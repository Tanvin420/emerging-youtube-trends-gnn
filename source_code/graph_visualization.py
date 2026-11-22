"""Recreates the original paper's "correlated graph" visualization (network
plot with node size scaled by centrality) on our corrected daily-snapshot
keyword-overlap graphs, one panel per region for the same calendar day."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from . import centrality, config, graph_construction, sis_labels


plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 13,
    "legend.fontsize": 11,
})


def plot_correlated_graphs(day: str, regions=config.REGIONS, out_name="fig0_correlated_graph.pdf"):
    fig, axes = plt.subplots(2, 2, figsize=(6.4, 5.6))
    for ax, region in zip(axes.flat, regions):
        graphs = graph_construction.build_and_cache(region)
        if day not in graphs:
            ax.set_visible(False)
            continue
        G = graphs[day]

        cent_df = sis_labels.add_top20_label(centrality.build_and_cache(region))
        day_cent = cent_df[cent_df["month"] == day].set_index("video_id")
        deg = day_cent["degree"].reindex(G.nodes()).fillna(0).to_numpy()

        pos = nx.spring_layout(G, seed=42, k=0.6 / np.sqrt(max(G.number_of_nodes(), 1)))
        node_sizes = 30 + 400 * deg
        node_colors = deg

        nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.15, width=0.5, edge_color="#8FBFE0")
        nodes = nx.draw_networkx_nodes(
            G, pos, ax=ax, node_size=node_sizes, node_color=node_colors,
            cmap="Blues", vmin=0, vmax=1, linewidths=0.4, edgecolors="#154360",
        )
        ax.set_title(f"{config.REGION_NAMES.get(region, region)} ({day})")
        ax.axis("off")

    fig.suptitle("Keyword-overlap graph, size/color = degree centrality", fontsize=13)
    fig.tight_layout(pad=0.4)
    fig.savefig(config.FIGURES / out_name, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    plot_correlated_graphs("2022-01-10")
    print("saved to", config.FIGURES / "fig0_correlated_graph.pdf")
