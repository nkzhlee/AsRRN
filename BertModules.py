import torch
import torch.nn as nn
from pytorch_transformers import BertModel
from constants import *

class BertClassifier(nn.Module):

    def __init__(self, config, DEVICE):
        super(BertClassifier, self).__init__()
        # Binary classification problem (num_labels = 2)
        self.device = DEVICE
        self.num_labels = config.num_labels
        # Pre-trained BERT model
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        # Graph
        self.n_nodes = hyperparameters['n_nodes']
        self.n_steps = hyperparameters['n_steps']
        #self.bert = BertModel(config)
        # Dropout to avoid overfitting
        self.dropout = nn.Dropout(hyperparameters["hidden_dropout_prob"])
        # A single layer classifier added on top of BERT to fine tune for binary classification
        self.mlp_hidden = hyperparameters['hidden_dim']
        self.o = nn.Sequential(
            #nn.Linear(self.mlp_hidden, mid_dim),
            #nn.ReLU(),
            #nn.Dropout(),
            #nn.Linear(mid_dim, NUM_LABELS),
            #nn.ReLU(),
            nn.Linear(self.mlp_hidden, hyperparameters['NUM_LABELS']),
        )
        # massage passing function f
        self.f = nn.Sequential(
            nn.Linear(2 * self.mlp_hidden, self.mlp_hidden),
            nn.ReLU(),
            nn.Linear(self.mlp_hidden, hyperparameters['mid_dim']),
            nn.ReLU(),
            nn.Dropout(),
        )
        # state update function 
        self.g = nn.Sequential(
            nn.Linear(2 * self.mlp_hidden + hyperparameters['mid_dim'], self.mlp_hidden),
            nn.ReLU(),
            # nn.Linear(self.mlp_hidden, self.mlp_hidden),
            # nn.ReLU(),
            nn.Dropout(),
        )

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                position_ids=None, head_mask=None):
        # Forward pass through pre-trained BERT
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)

        # Last layer output (Total 12 layers)
        pooled_output = outputs[-1]
        #print('pooled_output: ', pooled_output.size())
        pooled_output = self.dropout(pooled_output).unsqueeze(0)
        super_node = torch.randn(1, hyperparameters['hidden_dim']).unsqueeze(0).to(self.device)
        initial_state = torch.cat((super_node, pooled_output.clone()), dim=1)
        #print('initial_state: ', initial_state.size())
        states = initial_state.clone()  # record of states of all nodes
        # c_state = initial_state.clone() # should be (N, every node repeat n times,hidden dim)
        # h_state = initial_state.clone() # should be (All states repeat n times, N, hidden dim)
        for step in range(self.n_steps):
            # 2) create messages
            # a) copy every states n times, n = N
            c_state = states.permute((1, 0, 2))
            c_state = c_state.expand((-1, self.n_nodes + 1, -1))
            #print('c_state: ', c_state.size())
            h_state = states.expand((self.n_nodes + 1, -1, -1))
            #print('h_state: ', h_state.size())
            # b) concatenate h_i and h_j
            f_input = torch.cat((c_state, h_state), dim=2)
            # print('f_input: ', f_input.size())
            # c) m_ij = f(h_i, h_j)
            massages = self.f(f_input)
            massages = massages * mask.to(self.device)
            # print('massages: ', massages.size())
            # 3) message passing
            # a) sum up all the incoming messages
            massages = torch.sum(massages, 1)
            # print('massages: ', massages.size())
            # 4) update states
            # new states = g(states, initial states, massages)
            g_input = torch.cat((states.squeeze(), initial_state.squeeze(), massages), dim=1)
            # print('g_input: ', g_input.size())
            states = self.g(g_input)
            states = states.unsqueeze(0)
            # print('states: ', states.size())
        # remove the supernode from the node list
        answer_state = states.squeeze(0)[-1]
        # print('answer_state: ', answer_state.size())
        output = self.o(answer_state.unsqueeze(0))
        output = torch.sigmoid(output)
        # print('output: ', output.size())
        # assert 1 == 0
        return output, states.squeeze(0)
        # return self.classifier(pooled_output)[0].unsqueeze(0)
