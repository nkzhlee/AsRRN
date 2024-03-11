import torch
from transformers import AutoModel
from constants import *

# Define the sequence classification model
class AsRRN(torch.nn.Module):
    def __init__(self, DEVICE):
        super(AsRRN, self).__init__()
        self.device = DEVICE
        self.bert = AutoModel.from_pretrained(param['model_name'], local_files_only=True)
        self.dropout = torch.nn.Dropout(param['hidden_dropout_prob'])
        self.classifier = torch.nn.Linear(param['hidden_dim'], param['num_labels'])
        self.n_nodes = param['n_nodes']
        self.n_steps = param['n_steps']
        # self.bert = BertModel(config)
        # Dropout to avoid overfitting
        self.dropout = torch.nn.Dropout(param["hidden_dropout_prob"])
        # A single layer classifier added on top of BERT to fine tune for binary classification
        self.mlp_hidden = param['hidden_dim']
        self.o = torch.nn.Sequential(
            torch.nn.Linear(self.mlp_hidden, param['num_labels']),
        )
        # massage passing function f
        self.f = torch.nn.Sequential(
            torch.nn.Linear(2 * self.mlp_hidden, param['message_dim']),
            torch.nn.ReLU(),
            # nn.Linear(self.mlp_hidden, hyperparameters['mid_dim']),
            # nn.ReLU(),
            torch.nn.Dropout(param["hidden_dropout_prob"]),
        )

        self.p = torch.nn.Sequential(
            torch.nn.Linear(self.mlp_hidden * 2, self.mlp_hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(),
        )
        # state update function
        self.g = torch.nn.Sequential(
            torch.nn.Linear(2 * self.mlp_hidden + param['message_dim'], self.mlp_hidden),
            torch.nn.ReLU(),
            # nn.Linear(self.mlp_hidden, self.ml p_hidden),
            # nn.ReLU(),
            torch.nn.Dropout(param["hidden_dropout_prob"]),
        )
    def forward(self, input_ids, attention_mask=None):
        # print('input_ids: ', input_ids.size())
        # print('attention_mask: ', attention_mask.size())
        outputs = self.bert(input_ids.squeeze(), attention_mask=attention_mask.squeeze())

        pooled_output = outputs[-1]
        #print('pooled_output: ', pooled_output.size())
        pooled_output = self.dropout(pooled_output).unsqueeze(0)
        # print('part_2: ', part_2.size())
        super_node = torch.randn(1, param['hidden_dim']).unsqueeze(0).to(self.device)
        # print('super_node: ', super_node.size())
        initial_state = torch.cat((super_node, pooled_output), dim=1)
        # print('initial_state: ', initial_state.size())
        # assert 1 == 0
        states = initial_state.clone()  # record of states of all nodes
        # c_state = initial_state.clone() # should be (N, every node repeat n times,hidden dim)
        # h_state = initial_state.clone() # should be (All states repeat n times, N, hidden dim)
        for step in range(self.n_steps):
            # 2) create messages
            # a) copy every states n times, n = N
            c_state = states.permute((1, 0, 2))
            c_state = c_state.expand((-1, self.n_nodes + 1, -1))
            # print('c_state: ', c_state.size())
            h_state = states.expand((self.n_nodes + 1, -1, -1))
            # print('h_state: ', h_state.size())
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
            # 021
            #g_input = torch.cat((states.squeeze(), massages, initial_state.squeeze()), dim=1)
            g_input = torch.cat((states.squeeze(), initial_state.squeeze(), massages), dim=1)
            # print('g_input: ', g_input.size())
            states = self.g(g_input)
            states = states.unsqueeze(0)
            # print('states: ', states.size())
        # remove the supernode from the node list
        answer_state = states.squeeze(0)[-1]
        # print('answer_state: ', answer_state.size())
        output = self.o(answer_state.unsqueeze(0))
        logits = torch.softmax(output, dim=1)
        #logits = output
        return logits, states.squeeze(0)
