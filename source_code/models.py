"""Step 12-14: MLP / GCN / GAT / Centrality-Guided GAT (CGA-GAT) for
predicting future structural influence (binary node classification)."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import softmax


class CGAGATConv(nn.Module):
    """GAT layer whose attention logits are biased by a learnable projection
    of each neighbor's classical centrality vector (Sec. 13 of the plan):

        e_ij = a^T[Wh_i || Wh_j] + lambda * phi(c_j)
        alpha_ij = softmax_j(e_ij)
    """

    def __init__(self, in_channels, out_channels, heads=4, centrality_dim=4,
                 dropout=0.2, negative_slope=0.2, concat=True):
        super().__init__()
        self.heads = heads
        self.out_channels = out_channels
        self.concat = concat
        self.dropout = dropout
        self.negative_slope = negative_slope

        self.lin = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.att_src = nn.Parameter(torch.empty(1, heads, out_channels))
        self.att_dst = nn.Parameter(torch.empty(1, heads, out_channels))
        self.phi = nn.Linear(centrality_dim, heads)
        self.lambda_ = nn.Parameter(torch.tensor(1.0))
        self.bias = nn.Parameter(torch.zeros(heads * out_channels if concat else out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lin.weight)
        glorot(self.att_src)
        glorot(self.att_dst)
        glorot(self.phi.weight)
        zeros(self.phi.bias)
        zeros(self.bias)

    def forward(self, x, edge_index, cent):
        H, C = self.heads, self.out_channels
        N = x.size(0)

        x_proj = self.lin(x).view(N, H, C)
        alpha_src = (x_proj * self.att_src).sum(dim=-1)  # [N, H]
        alpha_dst = (x_proj * self.att_dst).sum(dim=-1)  # [N, H]
        p = self.phi(cent)  # [N, H] centrality prior per node

        src, dst = edge_index[0], edge_index[1]
        e = alpha_src[src] + alpha_dst[dst] + self.lambda_ * p[src]
        e = F.leaky_relu(e, self.negative_slope)
        alpha = softmax(e, dst, num_nodes=N)  # softmax over incoming edges per dst
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        msg = x_proj[src] * alpha.unsqueeze(-1)  # [E, H, C]
        out = torch.zeros(N, H, C, device=x.device, dtype=x.dtype)
        out.index_add_(0, dst, msg)

        if self.concat:
            out = out.view(N, H * C)
        else:
            out = out.mean(dim=1)
        return out + self.bias


class SPCGAGATConv(nn.Module):
    """Self-Preserving Centrality-Guided Graph Attention (SP-CGA-GAT).

    Root-cause fix for CGA-GAT's underperformance: CGAGATConv above has no
    self-loop and no dedicated self-transform, so a node's own
    representation is *fully replaced* by its neighbor aggregation at every
    layer -- unlike PyG's GATConv/GCNConv (which add self-loops by default,
    add_self_loops=True) or GraphSAGE (which keeps an explicit separate
    "root" weight for the node itself, root_weight=True by default). That
    asymmetry plausibly explains a good part of the CGA-GAT vs. GAT vs. SAGE
    gap observed empirically, not any fundamental weakness of attention
    itself. This layer adds a GraphSAGE-style dedicated self-transform
    alongside the centrality-guided attention aggregation over true
    neighbors (softmax excludes self, so neighbors don't have to compete
    with a self-loop for attention mass):

        h_i' = W_self h_i + sum_{j in N(i)} alpha_ij (W_neigh h_j)
        e_ij = a^T[W_neigh h_i || W_neigh h_j] + lambda * phi(c_j)
        alpha_ij = softmax_j(e_ij)
    """

    def __init__(self, in_channels, out_channels, heads=4, centrality_dim=4,
                 dropout=0.2, negative_slope=0.2, concat=True, use_centrality=True):
        super().__init__()
        self.heads = heads
        self.out_channels = out_channels
        self.concat = concat
        self.dropout = dropout
        self.negative_slope = negative_slope
        self.use_centrality = use_centrality

        self.lin_neigh = nn.Linear(in_channels, heads * out_channels, bias=False)
        out_total = heads * out_channels if concat else out_channels
        self.lin_self = nn.Linear(in_channels, out_total, bias=False)
        self.att_src = nn.Parameter(torch.empty(1, heads, out_channels))
        self.att_dst = nn.Parameter(torch.empty(1, heads, out_channels))
        if use_centrality:
            self.phi = nn.Linear(centrality_dim, heads)
            self.lambda_ = nn.Parameter(torch.tensor(1.0))
        self.bias = nn.Parameter(torch.zeros(out_total))
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lin_neigh.weight)
        glorot(self.lin_self.weight)
        glorot(self.att_src)
        glorot(self.att_dst)
        if self.use_centrality:
            glorot(self.phi.weight)
            zeros(self.phi.bias)
        zeros(self.bias)

    def forward(self, x, edge_index, cent):
        H, C = self.heads, self.out_channels
        N = x.size(0)

        x_neigh = self.lin_neigh(x).view(N, H, C)
        alpha_src = (x_neigh * self.att_src).sum(dim=-1)  # [N, H]
        alpha_dst = (x_neigh * self.att_dst).sum(dim=-1)  # [N, H]

        src, dst = edge_index[0], edge_index[1]
        e = alpha_src[src] + alpha_dst[dst]
        if self.use_centrality:
            p = self.phi(cent)  # [N, H] centrality prior per node
            e = e + self.lambda_ * p[src]
        e = F.leaky_relu(e, self.negative_slope)
        alpha = softmax(e, dst, num_nodes=N)  # softmax over true neighbors only
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        msg = x_neigh[src] * alpha.unsqueeze(-1)  # [E, H, C]
        agg = torch.zeros(N, H, C, device=x.device, dtype=x.dtype)
        agg.index_add_(0, dst, msg)
        agg = agg.view(N, H * C) if self.concat else agg.mean(dim=1)

        return self.lin_self(x) + agg + self.bias


class SPCGAGATModel(nn.Module):
    """Two SP-CGA-GAT layers + linear head. See SPCGAGATConv docstring.
    use_centrality=False drops the centrality-guided attention bias
    entirely, giving plain self-preserving attention (no centrality
    signal in the attention path at all, though centrality is still
    available elsewhere as a plain input feature) -- the missing rung in
    the GAT-to-GraphSAGE ladder: does self-preservation alone explain the
    gap, or is the centrality bias still doing something on top of it?"""

    def __init__(self, in_dim, hidden=128, heads=4, centrality_dim=4, dropout=0.2, use_centrality=True):
        super().__init__()
        self.conv1 = SPCGAGATConv(in_dim, hidden, heads=heads, centrality_dim=centrality_dim,
                                   dropout=dropout, concat=True, use_centrality=use_centrality)
        self.conv2 = SPCGAGATConv(hidden * heads, hidden, heads=heads, centrality_dim=centrality_dim,
                                   dropout=dropout, concat=False, use_centrality=use_centrality)
        self.head = nn.Linear(hidden, 1)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None, cent=None, edge_index_channel=None):
        x = F.elu(self.conv1(x, edge_index, cent))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv2(x, edge_index, cent))
        return self.head(x).squeeze(-1)


class _CentralityAttnHead(nn.Module):
    """One relation's centrality-guided attention aggregation -- the
    neighbor half of SPCGAGATConv, factored out so DualSPCGAGATConv can
    hold one instance per edge type and sum their aggregated messages
    before the shared self-transform, rather than forcing every relation
    to compete in one softmax."""

    def __init__(self, in_channels, out_channels, heads, centrality_dim,
                 dropout, negative_slope):
        super().__init__()
        self.heads = heads
        self.out_channels = out_channels
        self.dropout = dropout
        self.negative_slope = negative_slope
        self.lin_neigh = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.att_src = nn.Parameter(torch.empty(1, heads, out_channels))
        self.att_dst = nn.Parameter(torch.empty(1, heads, out_channels))
        self.phi = nn.Linear(centrality_dim, heads)
        self.lambda_ = nn.Parameter(torch.tensor(1.0))
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lin_neigh.weight)
        glorot(self.att_src)
        glorot(self.att_dst)
        glorot(self.phi.weight)
        zeros(self.phi.bias)

    def forward(self, x, edge_index, cent, num_nodes):
        H, C = self.heads, self.out_channels
        x_neigh = self.lin_neigh(x).view(num_nodes, H, C)

        if edge_index.numel() == 0:
            return torch.zeros(num_nodes, H, C, device=x.device, dtype=x.dtype)

        alpha_src = (x_neigh * self.att_src).sum(dim=-1)  # [N, H]
        alpha_dst = (x_neigh * self.att_dst).sum(dim=-1)  # [N, H]
        p = self.phi(cent)  # [N, H]

        src, dst = edge_index[0], edge_index[1]
        e = alpha_src[src] + alpha_dst[dst] + self.lambda_ * p[src]
        e = F.leaky_relu(e, self.negative_slope)
        alpha = softmax(e, dst, num_nodes=num_nodes)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        msg = x_neigh[src] * alpha.unsqueeze(-1)  # [E, H, C]
        agg = torch.zeros(num_nodes, H, C, device=x.device, dtype=x.dtype)
        agg.index_add_(0, dst, msg)
        return agg


class DualSPCGAGATConv(nn.Module):
    """SP-CGA-GAT extended with a second, structurally distinct edge type:
    same-channel co-trending videos (channel_graph.py), alongside the
    original keyword-overlap edges. Motivation: the deconfounded ablation
    (Sec. "Centrality's Role") found CGA-GAT's attention-bias term
    redundant once centrality is already a plain feature -- but that
    centrality is itself derived entirely from the keyword-overlap graph,
    so the finding says nothing about whether a *different* relational
    signal, invisible to that graph, would also be redundant. Creator
    identity (which channel a video is from) is exactly such a signal:
    Jaccard-over-TF-IDF-keywords cannot see it. Each edge type gets its own
    neighbor transform and centrality-guided attention head
    (_CentralityAttnHead); their aggregated messages are summed before the
    shared dedicated self-transform, so the two relations combine
    additively rather than competing for one softmax's attention mass --
    the same "don't force unlike things to compete for one softmax"
    principle that motivated keeping the self path separate from neighbor
    attention in SPCGAGATConv:

        h_i' = W_self h_i + sum_kw alpha^kw_ij (W^kw_neigh h_j)
                           + sum_ch alpha^ch_ij (W^ch_neigh h_j)

    use_keyword=False isolates the channel signal alone (an ablation arm
    mirroring the deconfounded-ablation style elsewhere in this project).
    """

    def __init__(self, in_channels, out_channels, heads=4, centrality_dim=4,
                 dropout=0.2, negative_slope=0.2, concat=True,
                 use_keyword=True, use_channel=True):
        super().__init__()
        assert use_keyword or use_channel, "DualSPCGAGATConv needs at least one edge type"
        self.heads = heads
        self.out_channels = out_channels
        self.concat = concat
        self.use_keyword = use_keyword
        self.use_channel = use_channel

        out_total = heads * out_channels if concat else out_channels
        self.lin_self = nn.Linear(in_channels, out_total, bias=False)
        if use_keyword:
            self.kw_head = _CentralityAttnHead(in_channels, out_channels, heads,
                                                centrality_dim, dropout, negative_slope)
        if use_channel:
            self.ch_head = _CentralityAttnHead(in_channels, out_channels, heads,
                                                centrality_dim, dropout, negative_slope)
        self.bias = nn.Parameter(torch.zeros(out_total))
        glorot(self.lin_self.weight)
        zeros(self.bias)

    def forward(self, x, edge_index, cent, edge_index_channel=None):
        N = x.size(0)
        H, C = self.heads, self.out_channels
        agg = torch.zeros(N, H, C, device=x.device, dtype=x.dtype)
        if self.use_keyword:
            agg = agg + self.kw_head(x, edge_index, cent, N)
        if self.use_channel:
            ei_ch = edge_index_channel
            if ei_ch is None:
                ei_ch = torch.zeros((2, 0), dtype=torch.long, device=x.device)
            agg = agg + self.ch_head(x, ei_ch, cent, N)
        agg = agg.view(N, H * C) if self.concat else agg.mean(dim=1)
        return self.lin_self(x) + agg + self.bias


class DualSPCGAGATModel(nn.Module):
    """Two DualSPCGAGATConv layers + linear head. See DualSPCGAGATConv
    docstring. use_keyword/use_channel select which edge type(s) are used,
    so this one class covers both the "keyword+channel" arm and the
    "channel-only" ablation arm of run_experiments.run_channel_augmentation_experiment."""

    def __init__(self, in_dim, hidden=128, heads=4, centrality_dim=4, dropout=0.2,
                 use_keyword=True, use_channel=True):
        super().__init__()
        self.conv1 = DualSPCGAGATConv(in_dim, hidden, heads=heads, centrality_dim=centrality_dim,
                                       dropout=dropout, concat=True,
                                       use_keyword=use_keyword, use_channel=use_channel)
        self.conv2 = DualSPCGAGATConv(hidden * heads, hidden, heads=heads, centrality_dim=centrality_dim,
                                       dropout=dropout, concat=False,
                                       use_keyword=use_keyword, use_channel=use_channel)
        self.head = nn.Linear(hidden, 1)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None, cent=None, edge_index_channel=None):
        x = F.elu(self.conv1(x, edge_index, cent, edge_index_channel))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv2(x, edge_index, cent, edge_index_channel))
        return self.head(x).squeeze(-1)


class MLPModel(nn.Module):
    def __init__(self, in_dim, hidden=128, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x, edge_index=None, edge_weight=None, cent=None, edge_index_channel=None):
        return self.net(x).squeeze(-1)


class GCNModel(nn.Module):
    def __init__(self, in_dim, hidden=128, dropout=0.2):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.head = nn.Linear(hidden, 1)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None, cent=None, edge_index_channel=None):
        x = F.elu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv2(x, edge_index, edge_weight))
        return self.head(x).squeeze(-1)


class GraphSAGEModel(nn.Module):
    """GraphSAGE (Hamilton et al., 2017) -- mean-aggregator message passing,
    included as a widely-used state-of-the-art GNN baseline alongside GCN/GAT."""

    def __init__(self, in_dim, hidden=128, dropout=0.2):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden)
        self.conv2 = SAGEConv(hidden, hidden)
        self.head = nn.Linear(hidden, 1)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None, cent=None, edge_index_channel=None):
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv2(x, edge_index))
        return self.head(x).squeeze(-1)


class GATModel(nn.Module):
    def __init__(self, in_dim, hidden=128, heads=4, dropout=0.2):
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden, heads=heads, dropout=dropout, concat=True)
        self.conv2 = GATConv(hidden * heads, hidden, heads=1, dropout=dropout, concat=False)
        self.head = nn.Linear(hidden, 1)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None, cent=None, edge_index_channel=None):
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv2(x, edge_index))
        return self.head(x).squeeze(-1)


class CGAGATModel(nn.Module):
    def __init__(self, in_dim, hidden=128, heads=4, centrality_dim=4, dropout=0.2):
        super().__init__()
        self.conv1 = CGAGATConv(in_dim, hidden, heads=heads, centrality_dim=centrality_dim,
                                 dropout=dropout, concat=True)
        self.conv2 = CGAGATConv(hidden * heads, hidden, heads=1, centrality_dim=centrality_dim,
                                 dropout=dropout, concat=False)
        self.head = nn.Linear(hidden, 1)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None, cent=None, edge_index_channel=None):
        x = F.elu(self.conv1(x, edge_index, cent))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv2(x, edge_index, cent))
        return self.head(x).squeeze(-1)


class ResidualGATModel(nn.Module):
    """GAT with a skip connection from the raw input features straight to
    the classification head (concatenated with the final graph
    representation), motivated by an empirical finding: plain GAT/CGA-GAT
    clearly underperform a feature-only MLP on the (rare-event) emergence
    task, suggesting message passing is diluting an already-strong
    self-feature signal rather than complementing it. The skip connection
    lets the model fall back on that self-signal while still exploiting
    relational information when it helps -- the same idea behind
    APPNP/JKNet/GCNII's residual connections for GNN over-smoothing/feature
    dilution, applied here to a shallow 2-layer GAT."""

    def __init__(self, in_dim, hidden=128, heads=4, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden)
        self.conv1 = GATConv(in_dim, hidden, heads=heads, dropout=dropout, concat=True)
        self.conv2 = GATConv(hidden * heads, hidden, heads=1, dropout=dropout, concat=False)
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None, cent=None, edge_index_channel=None):
        x0 = F.elu(self.input_proj(x))
        h = F.elu(self.conv1(x, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.elu(self.conv2(h, edge_index))
        combined = torch.cat([h, x0], dim=-1)
        return self.head(combined).squeeze(-1)


class ResidualCGAGATModel(nn.Module):
    """CGA-GAT with the same input-to-head skip connection as
    ResidualGATModel, so the centrality-guided attention mechanism gets the
    same chance to complement (rather than compete with) the self-feature
    signal."""

    def __init__(self, in_dim, hidden=128, heads=4, centrality_dim=4, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden)
        self.conv1 = CGAGATConv(in_dim, hidden, heads=heads, centrality_dim=centrality_dim,
                                 dropout=dropout, concat=True)
        self.conv2 = CGAGATConv(hidden * heads, hidden, heads=1, centrality_dim=centrality_dim,
                                 dropout=dropout, concat=False)
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None, cent=None, edge_index_channel=None):
        x0 = F.elu(self.input_proj(x))
        h = F.elu(self.conv1(x, edge_index, cent))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.elu(self.conv2(h, edge_index, cent))
        combined = torch.cat([h, x0], dim=-1)
        return self.head(combined).squeeze(-1)


def build_model(name: str, in_dim: int, centrality_dim: int = 4, hidden=128, heads=4, dropout=0.2):
    name = name.lower()
    if name == "mlp":
        return MLPModel(in_dim, hidden=hidden, dropout=dropout)
    if name == "gcn":
        return GCNModel(in_dim, hidden=hidden, dropout=dropout)
    if name == "sage":
        return GraphSAGEModel(in_dim, hidden=hidden, dropout=dropout)
    if name == "gat":
        return GATModel(in_dim, hidden=hidden, heads=heads, dropout=dropout)
    if name == "cga-gat":
        return CGAGATModel(in_dim, hidden=hidden, heads=heads, centrality_dim=centrality_dim, dropout=dropout)
    if name == "gat-res":
        return ResidualGATModel(in_dim, hidden=hidden, heads=heads, dropout=dropout)
    if name == "cga-gat-res":
        return ResidualCGAGATModel(in_dim, hidden=hidden, heads=heads, centrality_dim=centrality_dim, dropout=dropout)
    if name == "spcga-gat":
        return SPCGAGATModel(in_dim, hidden=hidden, heads=heads, centrality_dim=centrality_dim, dropout=dropout)
    if name == "sp-gat":
        return SPCGAGATModel(in_dim, hidden=hidden, heads=heads, centrality_dim=centrality_dim,
                              dropout=dropout, use_centrality=False)
    if name == "dual-spcga-gat":
        return DualSPCGAGATModel(in_dim, hidden=hidden, heads=heads, centrality_dim=centrality_dim,
                                  dropout=dropout, use_keyword=True, use_channel=True)
    if name == "spcga-gat-channelonly":
        return DualSPCGAGATModel(in_dim, hidden=hidden, heads=heads, centrality_dim=centrality_dim,
                                  dropout=dropout, use_keyword=False, use_channel=True)
    raise ValueError(f"unknown model {name}")
