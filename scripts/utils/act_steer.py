from steering_vectors import SteeringVector, train_steering_vector, extract_activations, record_activations
from typing import List
import joblib


def make_save_activations_structure(pos_acts_by_layer, neg_acts_by_layer, pairs):
    paired_data = []
    layer_ids = list(pos_acts_by_layer.keys())
    for i, pair_prompts in enumerate(pairs):

        pos_embeddings = {}
        neg_embeddings = {}

        for layer_num in layer_ids:
            pos_embeddings[layer_num] = pos_acts_by_layer[layer_num][i]
            neg_embeddings[layer_num] = neg_acts_by_layer[layer_num][i]

        paired_data.append({
            "pair_index": i,
            "prompts": pair_prompts,
            "pos": pos_embeddings,
            "neg": neg_embeddings
        })

    return paired_data


def get_steering_vector(model, tokenizer, data_pairs, layers: List, path_to_save: str, token_index=-2):
    data_steer = train_steering_vector(
        model,
        tokenizer,
        data_pairs,
        move_to_cpu=True,
        read_token_index=token_index,
        show_progress=True,
        layers=layers
    )

    joblib.dump(data_steer, path_to_save)
    print(f'save steering vectors for layers: {layers} to path: {path_to_save}')



def get_activations_from_pairs(model, tokenizer, data_pairs, path_to_save: str, layers, layer_type: str = 'self_attn'):
    if layers == -1:
        layers=list(range(model.config.num_hidden_layers))

    pos_acts_by_layer, neg_acts_by_layer = extract_activations(
        model=model,
        tokenizer=tokenizer,
        training_samples=data_pairs[:1000],
        layers=list(range(model.config.num_hidden_layers)),
        layer_type=layer_type,
        batch_size=1,
        show_progress=True,
        move_to_cpu=True
    )

    very_nervous_data = make_save_activations_structure(pos_acts_by_layer, neg_acts_by_layer, data_pairs)
    joblib.dump(very_nervous_data, path_to_save)

    print(f'save steering vectors for layers: {layers} to path: {path_to_save} with layer type: {layer_type}')
