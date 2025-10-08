//    Copyright 2025 时光丶人爱

//    Licensed under the Apache License, Version 2.0 (the "License");
//    you may not use this file except in compliance with the License.
//    You may obtain a copy of the License at

//        http://www.apache.org/licenses/LICENSE-2.0

//    Unless required by applicable law or agreed to in writing, software
//    distributed under the License is distributed on an "AS IS" BASIS,
//    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//    See the License for the specific language governing permissions and
//    limitations under the License.

#ifndef EXECUTOR_COMMAND_H
#define EXECUTOR_COMMAND_H
#include <optional>
#include <string>
#include <vector>

namespace north_star::executor {

std::string getToolPath(const std::string &tool);

struct Command {
  std::string _path;
  std::vector<std::string> _args;

  explicit Command(std::string exe_path);

  Command &appendStr(const std::string &arg);
  Command &appendStrOpt(const std::optional<std::string> &arg);
  Command &appendList(const std::vector<std::string> &args);
  Command &resetArgs();
  void exec(std::string wdir = "") const;
};

}  // namespace north_star::executor

#endif  // EXECUTOR_COMMAND_H
