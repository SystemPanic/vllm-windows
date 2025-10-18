import os
import sys

platform_h_file = os.path.join(sys.argv[1], "csrc", "cutlass", "include", "cutlass", "platform", "platform.h")

if os.path.exists(platform_h_file):
    with open(platform_h_file, mode="r", encoding="utf-8") as file:
        header_content = "".join(file.readlines())

    if "\n#if (201703L <=__cplusplus)\n" in header_content:
        header_content = header_content.replace("#if (201703L <=__cplusplus)", "#if defined(_MSC_VER) || (201703L <=__cplusplus)")
        with open(platform_h_file, mode="w", encoding="utf-8") as file:
            file.write(header_content)