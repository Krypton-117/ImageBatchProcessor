#pragma once
#include <filesystem>
#include <map>
#include <string>
struct AgentResult {
  std::filesystem::path source, recipe, preview, before;
  std::string job, error;
};
AgentResult agentGenerate(const std::filesystem::path&, const std::string&, bool teacher);
void agentFeedback(const std::string& job, bool accepted, const std::string& group);
void agentOpenPanel(const char* panel);
