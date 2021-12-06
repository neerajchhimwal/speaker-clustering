from resemblyzer import preprocess_wav, VoiceEncoder
from tqdm import tqdm
import glob
# from joblib import Parallel, delayed
import numpy as np
import math
import os


def audio_paths(directory):
    print('Using dir {}'.format(directory))
    return glob.glob(os.path.join(directory, '**/*.wav'), recursive=True)


def save_embeddings(embed_file_path, embeddings, file_paths):
    np.savez_compressed(embed_file_path, embeds=embeddings, file_paths=file_paths)
    print('Embeddings mapped to filepaths have been saved at {}'.format(embed_file_path))
    return


def encoder(file_paths, vocoder):
    print('Number of files in batch: {}'.format(len(file_paths)))

    # processed_wavs = Parallel(n_jobs=-1)(delayed(preprocess_wav)(i) for i in tqdm(file_paths))
    processed_wavs = [preprocess_wav(i) for i in tqdm(file_paths)]

    # encodings = Parallel(n_jobs=-1)(delayed(vocoder.embed_utterance)(i) for i in tqdm(processed_wavs))
    encodings = [vocoder.embed_utterance(i) for i in tqdm(processed_wavs)]
    print('Creating embeddings')
    encodings = np.array(encodings)
    return encodings


def concatenate_embed_files(embed_file_dest):
    pattern = '_*.npz'
    pattern_prefix = embed_file_dest[:-4]
    print(pattern_prefix)
    npz_files_to_concat = glob.glob(pattern_prefix + pattern, recursive=True)
    if npz_files_to_concat:
        print(npz_files_to_concat)
        list_of_loaded_files = []
        for file in npz_files_to_concat:
            list_of_loaded_files.append(np.load(file))

        final_embeds = np.concatenate([file['embeds'] for file in list_of_loaded_files])
        final_file_paths = np.concatenate([file['file_paths'] for file in list_of_loaded_files])
        print(f'Final length of concatenated embeds', len(final_embeds))
        save_embeddings(embed_file_dest, embeddings=final_embeds,
                        file_paths=final_file_paths)


def encode_on_partial_sets(source_dir, embed_file_path,
                           partial_set_size_for_embedding):
    vocoder = VoiceEncoder()
    file_paths = audio_paths(source_dir)
    print('Total number of files: {}'.format(len(file_paths)))
    if len(file_paths) <= partial_set_size_for_embedding:
        embeddings = encoder(file_paths=file_paths, vocoder=vocoder)
        save_embeddings(embed_file_path, embeddings, file_paths)
    else:
        for batch_no in range(math.ceil(len(file_paths) / partial_set_size_for_embedding)):
            start_index = batch_no * partial_set_size_for_embedding
            stop_index = partial_set_size_for_embedding * (batch_no + 1)
            batch_file_paths = file_paths[start_index:stop_index]
            embeddings = encoder(file_paths=batch_file_paths, vocoder=vocoder)
            new_embed_file_path = embed_file_path[:-4] + '_' + str(batch_no + 1) + '.npz'
            save_embeddings(new_embed_file_path, embeddings, batch_file_paths)

    concatenate_embed_files(embed_file_dest=embed_file_path)


if __name__ == "__main__":
    encode_on_partial_sets(source_dir='/path/to/folder/with/wavs/',
                           embed_file_path='source_embed_file.npz',
                           partial_set_size_for_embedding=100)
