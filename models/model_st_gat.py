import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class ST_GAT(torch.nn.Module):
    """
    Spatio-Temporal Graph Attention Network as presented https://ieeexplore.ieee.org/document/8903252
    """
    def __init__(self, in_channels, out_channels, n_nodes, heads=8, dropout=0):
        super(ST_GAT, self).__init__()
        self.n_nodes = n_nodes
        self.heads = heads
        self.dropout = dropout
        self.n_preds = out_channels

        lstm1_hidden_size = 32
        lstm2_hidden_size = 128

        # GAT layer
        self.gat = GATConv(in_channels=in_channels, out_channels=in_channels,
                           heads=heads, dropout=dropout, concat=False)
        
        # LSTM layers
        self.lstm1 = torch.nn.LSTM(input_size=self.n_nodes, hidden_size=lstm1_hidden_size, num_layers=1)
        for name, param in self.lstm1.named_parameters():
            if 'bias' in name:
                torch.nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                torch.nn.init.xavier_uniform_(param)
        
        self.lstm2 = torch.nn.LSTM(input_size=lstm1_hidden_size, hidden_size=lstm2_hidden_size, num_layers=1)
        for name, param in self.lstm2.named_parameters():
            if 'bias' in name:
                torch.nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                torch.nn.init.xavier_uniform_(param)
        
        # FC layer
        self.linear = torch.nn.Linear(lstm2_hidden_size, self.n_nodes*self.n_preds)
        torch.nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, data, device):
        x, edge_index = data.x, data.edge_index

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        x = torch.tensor(x, dtype=torch.float32, device=device)

        # GAT layer
        x = self.gat(x, edge_index)
        x = F.dropout(x, self.dropout, self.training)

        # LSTM layers
        batch_size = data.num_graphs
        n_nodes = int(data.num_nodes / batch_size)

        # [batch_size * n_nodes, seq_len]-> [batch_size, n_nodes, seq_len] 
        x = torch.reshape(x, (batch_size, n_nodes, data.num_features))

        # for lstm -> (seq_len, batch_size, n_nodes)
        x = torch.movedim(x, 2, 0)

        # [12, batch_size, 228] -> [12, batch_size, 32]
        x, _ = self.lstm1(x)

        # [12, batch_size, 32] -> [12, batch_size, 128]
        x, _ = self.lstm2(x)

        # output contains h_t for each timestep -> only the last one has all input's accounted for
        # [12, batch_size, 128] -> [50, 128]
        x = torch.squeeze(x[-1, :, :])

        # FC layer
        # [batch_size, 128] -> [batch_size, 228*9]
        x = self.linear(x)

        x_s = x.shape

        # [batch_size, 228*9] -> [batch_size, 228, 9]
        x = torch.reshape(x, (x_s[0], self.n_nodes, self.n_preds))

        # [batch_size, 228, 9] -> [batch_size*self]
        x = torch.reshape(x, (x_s[0]*self.n_nodes, self.n_preds))

        return x