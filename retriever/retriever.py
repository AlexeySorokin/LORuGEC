from functools import partial


import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModel

# from model_funcs import Detector

def calculate_cosine_distances(first, second):
    first_normalized = first / first.norm(dim=-1)[...,None]
    second_normalized = second / second.norm(dim=-1)[...,None]
    return 0.5*(1-torch.sum(first_normalized*second_normalized, dim=-1))


class EmbedderForPairwiseSimilarityRetrieval(nn.Module):

    def __init__(self, model: AutoModel, margin=None, temperature=1.0):
        super(EmbedderForPairwiseSimilarityRetrieval, self).__init__()
        self.model = model
        self.margin = margin
        self.temperature = temperature

    def _calculate_loss(self, positive_dist, negative_dist):
        gap = (positive_dist - negative_dist) / self.temperature
        if self.margin is None:
            loss = torch.log1p(torch.exp(gap))
        else:
            loss = nn.functional.relu(self.margin + gap)
        return loss.mean()

    def forward(self, input_ids, **kwargs):
        input_ids = input_ids.to(self.model.device)
        encoder_output = self.model(input_ids)
        encoder_output = encoder_output.pooler_output
        query_states, positive_anchor_states, negative_anchor_states = encoder_output.reshape(3, -1, encoder_output.shape[-1])
        positive_distances = calculate_cosine_distances(query_states, positive_anchor_states)
        negative_distances = calculate_cosine_distances(query_states, negative_anchor_states)
        loss = self._calculate_loss(positive_distances, negative_distances)
        return {"loss": loss, "positive_distances": positive_distances, "negative_distances": negative_distances}

class GectorForPairwiseSimilarityRetrieval(nn.Module):

    def __init__(self, detector, margin=None, temperature=1.0):
        super(GectorForPairwiseSimilarityRetrieval, self).__init__()
        self.detector = detector
        self.margin = margin
        self.temperature = temperature

    def _calculate_loss(self, positive_dist, negative_dist):
        dist = (positive_dist - negative_dist) / self.temperature
        if self.margin is None:
            loss = torch.log1p(torch.exp(dist))
        else:
            loss = nn.functional.relu(self.margin + dist)
        return loss.mean()

    def forward(self, input_ids, position_query_positive, position_query_negative,
                position_positive, position_negative, **kwargs):
        encoder_output = self.detector(input_ids, **kwargs, output_hidden_states=True)
        m = encoder_output["last_hidden_state"].shape[0] // 3
        pos_query_states = encoder_output["last_hidden_state"][::3][torch.arange(m), position_query_positive]
        neg_query_states = encoder_output["last_hidden_state"][::3][torch.arange(m), position_query_negative]
        positive_anchor_states = encoder_output["last_hidden_state"][1::3][torch.arange(m), position_positive]
        negative_anchor_states = encoder_output["last_hidden_state"][2::3][torch.arange(m), position_negative]
        positive_distances = calculate_cosine_distances(pos_query_states, positive_anchor_states)
        negative_distances = calculate_cosine_distances(neg_query_states, negative_anchor_states)
        loss = self._calculate_loss(positive_distances, negative_distances)
        return {"loss": loss, "positive_distances": positive_distances, "negative_distances": negative_distances}        