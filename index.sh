TARGET_DIR="${1:-.}"
if [[ ! -d "$TARGET_DIR" ]]
    echo "Error: Directory '$TARGET_DIR' does not exist"
    exit 1
fi
echo "Removing comments from all files in: $TARGET_DIR"
backup_dir="$TARGET_DIR/backup_$(date +
echo "Creating backup in: $backup_dir"
mkdir -p "$backup_dir"
find "$TARGET_DIR" -type f \
    -not -path "*/backup_*" \
    -not -name "*.jpg" -not -name "*.jpeg" -not -name "*.png" -not -name "*.gif" \
    -not -name "*.pdf" -not -name "*.zip" -not -name "*.tar*" -not -name "*.exe" \
    -not -name "*.so" -not -name "*.dll" -not -name "*.dylib" | while read -r file
    if file "$file" | grep -q "text\|ASCII\|UTF-8"
        echo "Processing: $file"
        backup_file="$backup_dir/${file
        mkdir -p "$(dirname "$backup_file")"
        cp "$file" "$backup_file"
        sed -i.bak '
            /^[[:space:]]*
            /^[[:space:]]*\/\
            /^[[:space:]]*\/\*/d
            /^[[:space:]]*\*/d
            /^[[:space:]]*
            /^[[:space:]]*
            /^[[:space:]]*
            s/[[:space:]]*
            s/[[:space:]]*\/\/.*$
            s/[[:space:]]*
            s/[[:space:]]*
            s/[[:space:]]*
            /^[[:space:]]*$/d
        ' "$file" && rm -f "$file.bak"
        sed -i.bak '
            /\/\*/,/\*\
