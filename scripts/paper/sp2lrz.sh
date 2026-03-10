#!/bin/bash
#
# modify_file.sh
#
# Cleans up a file by removing specific strings and replacing others,
# then renames it from <name>.sh to <name>_params.txt.
#
# Usage:
#   ./modify_file.sh -f <filename> [-d <replacement>]
#
# Arguments:
#   -f <filename>     (required) Path to the .sh file to modify.
#   -d <replacement>  (optional) String to replace occurrences of "spark" with.
#                     Defaults to "lrz" if not specified.
#
# What the script does:
#   - Deletes all occurrences of "python main_nep.py"
#   - Deletes all occurrences of "--compile"
#   - Replaces all occurrences of "spark" with the value of -d (default: "lrz")
#   - Renames the file: <name>.sh -> <name>_params.txt
#
# Examples:
#   ./modify_file.sh -f BPI17_001.sh
#   ./modify_file.sh -f BPI17_001.sh -d mycluster
#

DEST="lrz"

while getopts "f:d:" opt; do
  case $opt in
    f) FILE="$OPTARG" ;;
    d) DEST="$OPTARG" ;;
    *) echo "Usage: $0 -f <filename> [-d <replacement>]"; exit 1 ;;
  esac
done

if [[ -z "$FILE" ]]; then
  echo "Error: -f <filename> is required."
  echo "Usage: $0 -f <filename> [-d <replacement>]"
  exit 1
fi

if [[ ! -f "$FILE" ]]; then
  echo "Error: file '$FILE' not found."
  exit 1
fi

sed -i \
  -e 's|python main_nep\.py||g' \
  -e 's|--compile||g' \
  -e "s|spark|${DEST}|g" \
  "$FILE"

NEWFILE="${FILE%.sh}_params.txt"
mv "$FILE" "$NEWFILE"

echo "Done: '$FILE' updated and renamed to '$NEWFILE' (spark -> '${DEST}')."