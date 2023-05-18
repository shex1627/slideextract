from typing import List, Dict, Tuple

def combine_transcript_slides(transcripts, slide_indices: List, joiner: str = " "):
    """
    transcripts: list of transcript objects
    slide_indices: list of slide indices
    """
    slide_trans_dict = {}
    prev_time = 0.0
    for i, time in enumerate(slide_indices):
        if i < len(slide_indices)-1:
            start_time = slide_indices[i]
            next_time = slide_indices[i+1]
            slide_trans_dict[time] = [t['text'] for t in transcripts if t['start'] >= start_time and t['start'] < next_time]
        else:
            slide_trans_dict[time] = [t['text'] for t in transcripts if t['start'] >= prev_time]
        prev_time = time
    # join the text for every item in the dict
    slide_trans_dict = {index: joiner.join(lines) for index, lines in slide_trans_dict.items()}
    return slide_trans_dict

def reformat_paragraph(text: str, max_line_length: int=80) -> str:
    """
    Reformats a string into a paragraph with lines of reasonable length.

    Args:
        text (str): The input string to be reformatted.
        max_line_length (int): The maximum line length for each line.

    Returns:
        str: The reformatted paragraph.

    """
    words = text.split()
    lines = []
    current_line = ''
    current_length = 0

    for word in words:
        word_length = len(word)

        if current_length + word_length + len(current_line) <= max_line_length:
            current_line += ' ' + word
            current_length += word_length + 1
        else:
            lines.append(current_line.strip())
            current_line = word
            current_length = word_length

    if current_line:
        lines.append(current_line.strip())

    formatted_paragraph = ' '.join(lines)
    return formatted_paragraph