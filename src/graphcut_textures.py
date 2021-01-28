import argparse
import os

import cv2
import matplotlib.pyplot as plt
import maxflow
import networkx as nx
import numpy as np


class GraphCuts:

    def __init__(self, src, target, mask, save_graph=False):
        """
        Initialize the graph and computes the min-cut.
        :param src: image to be blended
        :param target: background image
        :param mask: manual mask with constrained pixels
        :param save_graph: if true, graph is saved
        """

        assert (src.shape == target.shape), \
            f"Source and target dimensions must be same: {str(src.shape)} != {str(target.shape)}"

        # Creating the graph and adding nodes
        graph = maxflow.Graph[float]()
        node_ids = graph.add_grid_nodes((src.shape[0], src.shape[1]))

        self.compute_edge_weights(src, target)  # self.edge_weights is inside func(compute_edge_weights)

        # Adding non-terminal edges
        patch_height = src.shape[0]
        patch_width = src.shape[1]
        for row_idx in range(patch_height):
            for col_idx in range(patch_width):
                # Horizontal edge
                if col_idx + 1 < patch_width:
                    weight = self.edge_weights[row_idx, col_idx, 0]
                    graph.add_edge(node_ids[row_idx][col_idx],
                                   node_ids[row_idx][col_idx + 1],
                                   weight,
                                   weight)

                # Vertical edge
                if row_idx + 1 < patch_height:
                    weight = self.edge_weights[row_idx, col_idx, 1]
                    graph.add_edge(node_ids[row_idx][col_idx],
                                   node_ids[row_idx + 1][col_idx],
                                   weight,
                                   weight)

                # Adding terminal edge capacities for the pixels constrained to belong to the source/sink.
                # http://pmneila.github.io/PyMaxflow/maxflow.html
                # 검토) add_tedge 대신 다른 api 쓸 순 없을까? np.inf 넣기 싫은데.
                if np.array_equal(mask[row_idx, col_idx, :], [0, 255, 255]):
                    graph.add_tedge(node_ids[row_idx][col_idx], 0, np.inf)
                elif np.array_equal(mask[row_idx, col_idx, :], [255, 128, 0]):
                    graph.add_tedge(node_ids[row_idx][col_idx], np.inf, 0)

        # Plot graph
        if save_graph:
            nxg = graph.get_nx_graph()
            self.plot_graph_2d(nxg, (patch_height, patch_width))
            # 디버깅
            # print('nxg {}'.format(nxg)) # nxg
            # print('type of nxg {}'.format(type(nxg))) # type of nxg <class 'networkx.classes.digraph.DiGraph'>

        # Computing maxflow / mincut
        flow = graph.maxflow()
        self.sgm = graph.get_grid_segments(node_ids)

    def compute_edge_weights(self, src, target):
        """
        Compute edge weights based on matching quality cost.
        :param src: image to be blended (foreground)
        :param target: background image
        """

        self.edge_weights = np.zeros((src.shape[0], src.shape[1], 2))

        # Create shifted versions of the matrics for vectorized operations.
        src_left_shifted = np.roll(src, -1, axis=1)
        target_left_shifted = np.roll(target, -1, axis=1)
        src_up_shifted = np.roll(src, -1, axis=0)
        target_up_shifted = np.roll(target, -1, axis=0)

        eps = 1e-10  # Numerical stability

        # Horizontal weights
        horizontal_weight = np.sum(np.square(src - target, dtype=np.float) +
                                   np.square(src_left_shifted - target_left_shifted, dtype=np.float),
                                   axis=2)

        horizontal_norm_factor = np.sum(np.square(src - src_left_shifted, dtype=np.float) +
                                        np.square(target - target_left_shifted, dtype=np.float),
                                        axis=2)

        self.edge_weights[:, :, 0] = horizontal_weight / (horizontal_norm_factor + eps)

        # Vertical weights
        vertical_weight = np.sum(np.square(src - target, dtype=np.float) +
                                 np.square(src_up_shifted - target_up_shifted, dtype=np.float),
                                 axis=2)

        vertical_norm_factor = np.sum(np.square(src - src_up_shifted, dtype=np.float) +
                                      np.square(target - target_up_shifted, dtype=np.float),
                                      axis=2)

        self.edge_weights[:, :, 1] = vertical_weight / (vertical_norm_factor + eps)

    def plot_graph_2d(self, graph, nodes_shape,
                      plot_weights=True,
                      plot_terminals=True,
                      font_size=7):
        """
        Plot the graph to be used in graph cuts
        :param graph: Maxflow graph
        :param nodes_shape: patch shape
        :param plot_weights: if true, edge weights are shown
        :param plot_terminals: if true, the terminal nodes are shown
        :param font_size: text font size
        """

        X, Y = np.mgrid[:nodes_shape[0], :nodes_shape[1]]
        aux = np.array([Y.ravel(), X[::-1].ravel()]).T
        positions = {i: v for i, v in enumerate(aux)}
        positions['s'] = (-1, nodes_shape[0] / 2.0 - 0.5)
        positions['t'] = (nodes_shape[1], nodes_shape[0] / 2.0 - 0.5)

        # nx.draw(graph, cmap=plt.get_cmap('jet')) maxflow로 안 가져오고 networkx에서 바로 그리기
        plt.show()
        nxgraph = graph.get_nx_graph()
        print("nxgraph created")
        if not plot_terminals:
            nxgraph.remove_nodes_from(['s', 't'])

        plt.clf()
        nx.draw(nxgraph, pos=positions)

        if plot_weights:
            edge_labels = {}
            for u, v, d in nxgraph.edges(data=True):
                edge_labels[(u, v)] = d['weight']
            nx.draw_networkx_edge_labels(nxgraph,
                                         pos=positions,
                                         edge_labels=edge_labels,
                                         label_pos=0.3,
                                         font_size=font_size)

        plt.axis('equal')
        plt.show()

    def blend(self, src, target):
        """
        Blends the target image with the source image based on the graph cut.
        :param src: Source image
        :param target: Target image
        :return target : Blended image
        """
        target[self.sgm] = src[self.sgm]

        return target


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', dest='image_dir', required=True, help='Saved Path of Source & Target Images.')
    args = parser.parse_args()

    # Read the images and the mask.
    image_dir = args.image_dir
    src = cv2.imread(os.path.join(image_dir, 'src.jpg'))
    target = cv2.imread(os.path.join(image_dir, 'target.jpg'))
    mask = cv2.imread(os.path.join(image_dir, 'mask.png'))

    # Compute the min-cut.
    graphcuts = GraphCuts(src, target, mask)

    # Save the output.
    target = graphcuts.blend(src, target)
    cv2.imwrite(os.path.join(image_dir, "result.png"), target)
