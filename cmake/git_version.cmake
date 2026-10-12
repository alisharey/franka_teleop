# Writes git_version.h with the source tree's commit and dirty flag. Runs on
# every build (see CMakeLists.txt) but only touches the header when it changes,
# so recordings always carry the commit the running binary was built from.
execute_process(COMMAND git -C "${SOURCE_DIR}" rev-parse HEAD
                OUTPUT_VARIABLE commit OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
execute_process(COMMAND git -C "${SOURCE_DIR}" status --porcelain --untracked-files=no
                OUTPUT_VARIABLE status OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
if(NOT commit)
  set(commit "unknown")
endif()
if(status)
  set(dirty "true")
else()
  set(dirty "false")
endif()
set(content "#pragma once\ninline constexpr const char* kGitCommit = \"${commit}\";\ninline constexpr bool kGitDirty = ${dirty};\n")
if(EXISTS "${OUTPUT}")
  file(READ "${OUTPUT}" existing)
endif()
if(NOT existing STREQUAL content)
  file(WRITE "${OUTPUT}" "${content}")
endif()
