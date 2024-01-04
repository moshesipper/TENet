# copyright 2024 moshe sipper
# www.moshesipper.com

from pycocotools.coco import COCO

# List of common conjunctions and transitional words
remove_words = {
    'and', 'but', 'or', 'nor', 'although', 'because', 'while', 'if', 'unless', 'however', 'nevertheless', 'on', 'the', 'other', 'of', 'a', 'in', 'is', 'to', 'with', 'are', 'for', "it's", 'that', 'has', 'from'
}
remove_chars = str.maketrans('', '', ',".')  # remove ',', '"', '.' from words


def build_vocab(path2annot, vocab_size=200, filename=None):
    # path2annot: Path to the COCO annotations file
    coco = COCO(path2annot) # Initialize COCO object
    
    image_ids = coco.getImgIds() # Get all image IDs
    print(f'number of annotated images: {len(image_ids)}')
    # number of annotated images: 118287

    rawvocab = dict()

    # Loop through each image ID
    for image_id in image_ids:
        # Get annotations for the current image
        annotations = coco.loadAnns(coco.getAnnIds(imgIds=image_id))

        # Extract captions and append to the list
        captions = [annotation['caption'] for annotation in annotations]

        words = [word for caption in captions for word in caption.translate(remove_chars).split()]
        words = {word.lower() for word in words}
        for w in words:
            if w in rawvocab:
                rawvocab[w] += 1
            else:
                rawvocab[w] = 1

    count3 = sum(1 for count in rawvocab.values() if count <= 3)
    count1 = sum(1 for count in rawvocab.values() if count == 1)
    len1 = sum(1 for word in rawvocab.keys() if len(word) == 1)
    len2 = sum(1 for word in rawvocab.keys() if len(word) == 2)
    print(f'raw vocab {len(rawvocab)}, count3 {count3}, count1 {count1}, len1 {len1}, len2 {len2}')
    # raw vocab 30567, count3 19362, count1 14065, len1 48, len2 306

    vocab = dict()
    for word, count in rawvocab.items():
        if len(word) >= 3 and count >= 4 and word not in remove_words:
            vocab[word] = count

    vocab = dict(sorted(vocab.items(), key=lambda item: item[1], reverse=True))
    # import json
    # with open('v.json', 'w') as json_file:
    #     json.dump(vocab, json_file)
    top_vocab = list(vocab.keys())[:vocab_size]  # vocab_size most frequent words

    if filename is not None:
        with open(filename, 'w') as file:
            file.write('VOCAB = [')
            for idx, item in enumerate(top_vocab):
                if idx < len(top_vocab) - 1:
                    file.write(f"'{item}', ")
                else:
                    file.write(f"'{item}']")

    return top_vocab
