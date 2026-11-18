// KV-13 (Phase 6, machinery re-proof): continuation snapshots and
// rollback over the quantized layout. Until KV-13 the snapshot
// machinery sized every cache row binary16 (engine.cpp VisitBody /
// LayoutOf / SnapshotBytes), so a q8_0 session could not be
// snapshotted at all. The machinery now sizes rows in the cache's
// stored format (KvCacheRowBytes per layer). This test proves, for
// f16 (control, unchanged figures) and for q8_0:
//   - a snapshot taken after the sliding ring wrapped restores into
//     fresh sessions, typed and serialized, and continues bitwise
//     like the original for 8 greedy steps;
//   - rollback: restoring the payload into the advanced original
//     returns its exact snapshot-point logits, and a snapshot of the
//     restored session reproduces the original payload byte for byte;
//   - cross-format refusal: a payload or typed snapshot of one format
//     is refused by a session of the other, by name or by the
//     format-sensitive size check -- never silently reinterpreted.
// The MTP draft chain itself is not exercised here: it needs the
// separate MTP draft GGUF (GUFO_GEMMA4_MTP_MODEL), which this machine
// does not hold -- the same gap target_test records for f16.
// Requires GUFO_GEMMA4_MODEL; exits 77 without it.
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <gufo/ext/gguf_reader.hpp>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "chat_template.hpp"
#include "engine.hpp"
#include "tests/check.hpp"

namespace g4 = gufo::models::gemma4;
using gemma4_test::Require;

namespace {

std::vector<g4::TokenId> PromptTokens(g4::Model& model,
                                      const std::string& text) {
  gufo::ext::ChatMessage message(gufo::ext::ChatRole::kUser, text);
  std::string error;
  const auto rendered =
      g4::ChatTemplate::Render(std::span(&message, 1), {}, {}, &error);
  Require(rendered.has_value(), error);
  return model.Tokenize(rendered->text);
}

std::vector<float> AsVector(std::span<const float> v) {
  return std::vector<float>(v.begin(), v.end());
}

struct Proof {
  std::vector<std::uint8_t> payload;
  std::unique_ptr<g4::SessionSnapshot> snapshot;
  std::uint64_t snapshot_bytes{0};
};

Proof RunProof(g4::Model& model, const char* name) {
  std::string error;
  std::string text;
  for (int i = 0; i < 90; ++i) {
    text += "Ship " + std::to_string(i) +
            " left Genoa at dawn carrying salt, wine and letters. ";
  }
  const auto prompt = PromptTokens(model, text);
  Require(prompt.size() > model.SessionRingSlots() + 64 &&
              prompt.size() + 16 < 3072,
          "snapshot prompt does not wrap the sliding ring");
  auto original = model.CreateSession(0, &error);
  Require(original && original->Sync(prompt, &error), error);
  Proof proof;
  proof.snapshot = original->SaveSnapshot(&error);
  Require(proof.snapshot != nullptr, error);
  proof.snapshot_bytes = original->SnapshotBytes();
  Require(proof.snapshot->SizeBytes() == proof.snapshot_bytes,
          std::string("snapshot size mismatch (") + name + ")");
  proof.payload.resize(proof.snapshot->SizeBytes());
  Require(proof.snapshot->CopyTo(proof.payload), "payload copy failed");
  std::vector<std::uint8_t> streamed;
  Require(proof.snapshot->Stream([&](std::span<const std::uint8_t> piece) {
    streamed.insert(streamed.end(), piece.begin(), piece.end());
  }) && streamed == proof.payload,
          "streamed snapshot payload differs from its copy");
  auto typed = model.CreateSession(0, &error);
  auto serialized = model.CreateSession(0, &error);
  Require(typed && typed->RestoreSnapshot(*proof.snapshot, &error), error);
  Require(serialized && serialized->RestoreSnapshot(proof.payload, &error),
          error);
  const auto point_logits = AsVector(original->Logits());
  for (int step = 0; step <= 8; ++step) {
    const auto logits = AsVector(original->Logits());
    Require(AsVector(typed->Logits()) == logits &&
                AsVector(serialized->Logits()) == logits,
            std::string("restored session diverges (") + name + ") at step " +
                std::to_string(step));
    if (step == 8) {
      break;
    }
    const auto next = static_cast<g4::TokenId>(
        std::max_element(logits.begin(), logits.end()) - logits.begin());
    Require(original->Evaluate(next, &error) && typed->Evaluate(next, &error) &&
                serialized->Evaluate(next, &error),
            error);
  }
  // Rollback: the original has advanced 8 tokens past the snapshot;
  // restoring its own payload returns the snapshot point exactly.
  Require(original->RestoreSnapshot(proof.payload, &error), error);
  Require(AsVector(original->Logits()) == point_logits,
          std::string("rollback logits differ (") + name + ")");
  const auto again = original->SaveSnapshot(&error);
  Require(again != nullptr, error);
  std::vector<std::uint8_t> again_payload(again->SizeBytes());
  Require(again->CopyTo(again_payload) && again_payload == proof.payload,
          std::string("restored session's snapshot payload differs (") + name +
              ")");
  std::cout << "snapshot proof " << name << ": " << proof.snapshot_bytes
            << " bytes, bitwise continuation + rollback\n";
  return proof;
}

}  // namespace

int main() {
  const char* path = std::getenv("GUFO_GEMMA4_MODEL");
  if (path == nullptr || *path == '\0') {
    std::cout << "SKIP: GUFO_GEMMA4_MODEL is not set\n";
    return 77;
  }
  return gemma4_test::Run([&] {
    std::string error;
    const auto load = [&](g4::KvCacheFormat format) {
      g4::ModelOptions options;
      options.max_context = 3072;
      options.prefill_chunk = 240;
      options.kv_cache_format = format;
      auto model = g4::Model::Load(path, options, &error);
      Require(model != nullptr, error);
      return model;
    };
    Proof f16;
    {
      auto model = load(g4::KvCacheFormat::kF16);
      f16 = RunProof(*model, "f16");
    }
    Proof q8;
    {
      auto model = load(g4::KvCacheFormat::kQ8_0);
      q8 = RunProof(*model, "q8_0");
      Require(q8.snapshot_bytes < f16.snapshot_bytes,
              "q8_0 snapshot is not smaller than the f16 one");
      auto session = model->CreateSession(0, &error);
      Require(session != nullptr, error);
      std::string refused;
      Require(!session->RestoreSnapshot(f16.payload, &refused),
              "f16 payload accepted by a q8_0 session");
      Require(!session->RestoreSnapshot(*f16.snapshot, &refused) &&
                  refused.find("format") != std::string::npos,
              "f16 snapshot not refused by name in a q8_0 session: " + refused);
    }
    {
      auto model = load(g4::KvCacheFormat::kF16);
      auto session = model->CreateSession(0, &error);
      Require(session != nullptr, error);
      std::string refused;
      Require(!session->RestoreSnapshot(q8.payload, &refused),
              "q8_0 payload accepted by an f16 session");
      Require(
          !session->RestoreSnapshot(*q8.snapshot, &refused) &&
              refused.find("format") != std::string::npos,
          "q8_0 snapshot not refused by name in an f16 session: " + refused);
    }
    std::cout << "cross-format refusal: both directions, both paths\n";
  });
}
