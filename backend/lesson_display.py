"""Keep internal source markers out of learner-facing prose."""
import re


# Only known editorial annotations, never arbitrary angle brackets or parentheses.
EDITORIAL_MARKER = re.compile(r'[ \t]*(?:<출제됨>|&lt;출제됨&gt;)')


def display_title(text):
    return EDITORIAL_MARKER.sub('', text).strip()


def learner_text(text):
    # Preserve code spans/blocks and indexing expressions such as values[1].
    parts = re.split(r'(```[\s\S]*?```|`[^`\n]*`)', text)
    marker = r'(?<![A-Za-z0-9_\]])\[(?:\d+)(?:\s*[,，]\s*\d+)*\](?![\w\[(])'
    for i in range(0, len(parts), 2):
        parts[i] = EDITORIAL_MARKER.sub('', parts[i])
        parts[i] = re.sub(marker, '', parts[i])
        parts[i] = re.sub(r'[ \t]+([.,!?。])', r'\1', parts[i])
    return ''.join(parts)
