#!/bin/bash

# Script to remove comments and docstrings from all files in repository
# Usage: ./remove_comments.sh [directory]

TARGET_DIR="${1:-.}"

if [[ ! -d "$TARGET_DIR" ]]; then
    echo "Error: Directory '$TARGET_DIR' does not exist"
    exit 1
fi

echo "Removing comments from all files in: $TARGET_DIR"

# Create backup
backup_dir="$TARGET_DIR/backup_$(date +%Y%m%d_%H%M%S)"
echo "Creating backup in: $backup_dir"
mkdir -p "$backup_dir"

# Process all text files (skip binary files, .git, node_modules, etc.)
find "$TARGET_DIR" -type f \
    -not -path "*/.git/*" \
    -not -path "*/node_modules/*" \
    -not -path "*/backup_*" \
    -not -path "*/.venv/*" \
    -not -path "*/__pycache__/*" \
    -not -name "*.jpg" -not -name "*.jpeg" -not -name "*.png" -not -name "*.gif" \
    -not -name "*.pdf" -not -name "*.zip" -not -name "*.tar*" -not -name "*.exe" \
    -not -name "*.so" -not -name "*.dll" -not -name "*.dylib" | while read -r file; do
    
    # Check if file is text
    if file "$file" | grep -q "text\|ASCII\|UTF-8"; then
        echo "Processing: $file"
        
        # Create backup
        backup_file="$backup_dir/${file#$TARGET_DIR/}"
        mkdir -p "$(dirname "$backup_file")"
        cp "$file" "$backup_file"
        
        # Remove comments using universal patterns
        sed -i.bak '
            # Remove lines that are only comments
            /^[[:space:]]*#/d
            /^[[:space:]]*\/\//d
            /^[[:space:]]*\/\*/d
            /^[[:space:]]*\*/d
            /^[[:space:]]*--/d
            /^[[:space:]]*%/d
            /^[[:space:]]*;/d
            
            # Remove inline comments (basic patterns)
            s/[[:space:]]*#.*$//
            s/[[:space:]]*\/\/.*$//
            s/[[:space:]]*--.*$//
            s/[[:space:]]*%.*$//
            s/[[:space:]]*;.*$//
            
            # Remove empty lines
            /^[[:space:]]*$/d
        ' "$file" && rm -f "$file.bak"
        
        # Remove multi-line comment blocks
        sed -i.bak '
            /\/\*/,/\*\//d
            /"""/,/"""/d
            /'"'"''"'"''"'"'/,/'"'"''"'"''"'"'/d
            /<!--/,/-->/d
            /=begin/,/=end/d
        ' "$file" && rm -f "$file.bak"
    fi
done

echo "Processing complete!"
echo "Backup created at: $backup_dir"
echo "WARNING: This script removes comments using basic patterns."
echo "Review changes carefully before committing!"