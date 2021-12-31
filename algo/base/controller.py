import copy
import numpy as np
import networkx as nx
from itertools import islice

from .recorder import Solution
from data.physical_network import PhysicalNetwork

'''
Methods List

# 
bfs_deploy
undo_deploy: release and reset the solution
release

##
node_mapping
link_mapping
place_and_route
undo_place_and_route

###
can_place
can_route
place
route
undo_place
undo_route
'''

class Controller:
    
    @classmethod
    def can_place(cls, vn, pn, vid, pid) -> bool:
        """Check node constraints."""
        assert pid in list(pn.nodes)
        for n_attr in vn.get_node_attrs():
            if not n_attr.check(vn, pn, vid, pid):
                # FAILURE
                return False
        # SUCCESS
        return True

    @classmethod
    def can_route(cls, vn, pn, vn_link, pn_path) -> bool:
        """Check link constraints."""
        pn_edges = cls.path_to_edges(pn_path)
        for pn_edge in pn_edges:
            for e_attr in vn.edge_attrs:
                if not e_attr.check(vn, pn, vn_link, pn_edge):
                    return False
        return True

    @classmethod
    def place(cls, vn, pn, vid, pid, solution=None):
        r"""Attempt to place the VNF `vid` in PN node `pid`."""
        if not cls.can_place(vn, pn, vid, pid):
            # FAILURE
            return False
        # update node
        for n_attr in vn.get_node_attrs('resource'):
            n_attr.update(vn.nodes[vid], pn.nodes[pid], '-')

        if solution is not None: solution['node_slots'][vid] = pid
        return True

    @classmethod
    def route(cls, vn, pn, vn_link, pl_pair, solution=None, available_network=False, shortest_method='all_shortest', k=1):
        r"""Return True if route successfully the virtual link (vid_a, vid_b) in the physical network path (pid_a, pid_b); Otherwise False.
        
        Returns:
            route_result (bool): True if route successfully else False.
        """
        # place the prev VNF and curr VNF on the identical physical node
        if pl_pair[0] == pl_pair[1]:
            return True
        
        temp_network = cls.create_available_graph(vn, pn, vn_link) if available_network else pn

        shortest_paths = cls.find_shortest_paths(temp_network, pl_pair[0], pl_pair[1], method=shortest_method, k=k)
        for pn_path in shortest_paths:
            if available_network or cls.can_route(vn, pn, vn_link, pn_path):
                # SUCCESS
                for e_attr in vn.get_edge_attrs('resource'):
                    e_attr.update_path(vn.edges[vn_link], pn, pn_path, method='-')
                    solution['edge_paths'][vn_link] = pn_path
                if solution is not None: solution['edge_paths'][vn_link] = pn_path
                return True
        # FAILURE
        return False

    @classmethod
    def place_and_route(cls, vn, pn, vid, pid, solution, available_network=False, shortest_method='all_shortest', k=1) -> bool:
        r"""Attempt to place the VNF `vid` in PN node`pid` 
            and route VLs related to the VNF.
        """
        # Place
        place_result = cls.place(vn, pn, vid, pid, solution)
        if not place_result:
            # FAILURE
            solution['place_result'] = False
            return False

        # Route
        vid_neighbors = list(vn.adj[vid])
        for vid_nb in vid_neighbors:
            placed = vid_nb in solution['node_slots'].keys()
            routed = (vid_nb, vid) in solution['edge_paths'].keys() or (vid, vid_nb) in solution['edge_paths'].keys()
            if placed and not routed:
                n_pid = solution['node_slots'][vid_nb]
                route_result = cls.route(vn, pn, (vid, vid_nb), (pid, n_pid), solution, 
                                            available_network=available_network, shortest_method=shortest_method, k=k)
                if not route_result:
                    # FAILURE
                    solution['route_result'] = False
                    return False
        # SUCCESS
        return True

    # @classmethod
    # def can_place_and_route(cls, vn, pn, vid, pid, solution):
    #     # Place
    #     can_place_result = cls.can_place(vn, pn, vid, pid)
    #     if not can_place_result:
    #         return False
    #     # Route
    #     vid_neighbors = list(vn.adj[vid])
    #     for vid_nb in vid_neighbors:
    #         placed = vid_nb in solution['node_slots'].keys()
    #         routed = (vid_nb, vid) in solution['edge_paths'].keys() or (vid, vid_nb) in solution['edge_paths'].keys()
    #         if placed and not routed:
    #             pid_nb = solution['node_slots'][vid_nb]
    #             route_result = cls.can_route(vn, pn, (vid, vid_nb), (pid, pid_nb))
    #             if not route_result:
    #                 return False
    #     return True

    @classmethod
    def undo_place(cls, vn, pn, vid, pid, solution=None):
        for n_attr in vn.get_node_attrs('resource'):
            n_attr.update(vn.nodes[vid], pn.nodes[pid], '+')
        if solution is not None: del solution['node_slots'][vid]
        return True

    @classmethod
    def undo_route(cls, vn, pn, vn_link, path, solution=None):
        for e_attr in vn.get_edge_attrs('resource'):
            e_attr.update_path(vn.edges[vn_link], pn, path, method='+')
        if solution is not None: del solution['edge_paths'][vn_link]
        return True

    @classmethod
    def undo_place_and_route(cls, vn, pn, vid, pid, solution):
        # Undo place
        origin_node_slots = list(solution['node_slots'].keys())
        if vid not in origin_node_slots:
            return True
        undo_place_result = cls.undo_place(vn, pn, vid, pid, solution)
        # Undo route
        origin_node_slots = list(solution['edge_paths'].keys())
        for vn_link in origin_node_slots:
            if vid in vn_link:
                undo_route_result = cls.undo_route(vn, pn, vn_link, solution['edge_paths'][vn_link], solution)
        return True

    @classmethod
    def undo_deploy(cls, vn, pn, solution):
        r"""Release occupied resources when a VN leaves PN, and reset the solution."""
        cls.release(vn, pn, solution)
        solution.reset()
        return True

    @classmethod
    def bfs_deploy(cls, vn, pn, ordered_vn_nodes, pn_initial_node, max_visit=100, max_depth=10, 
                        available_network=False, shortest_method='all_shortest', k=1):
        r"""Deploy the `vn` in `pn` starting from `initial_node` using Breadth-First Search algorithm."""
        solution = Solution(vn)

        max_visit_in_every_depth = int(np.power(max_visit, 1 / max_depth))
        
        curr_depth = 0
        visited = pn.num_nodes * [False]
        queue = [(pn_initial_node, curr_depth)]
        visited[pn_initial_node] = True

        num_placed_nodes = 0
        vid = ordered_vn_nodes[num_placed_nodes]

        while queue:
            (curr_pid, depth) = queue.pop(0)
            if depth > max_depth:
                break

            if cls.place_and_route(vn, pn, vid, curr_pid, solution, 
                                    available_network=available_network, shortest_method=shortest_method, k=k):
                num_placed_nodes = num_placed_nodes + 1

                if num_placed_nodes >= len(ordered_vn_nodes):
                    solution['result'] = True
                    return solution
                vid = ordered_vn_nodes[num_placed_nodes]
            else:
                cls.undo_place_and_route(vn, pn, vid, curr_pid, solution)

            if depth == max_depth:
                continue

            node_edges = pn.edges(curr_pid, data=True)
            node_edges = node_edges if len(node_edges) <= max_visit else node_edges[:max_visit_in_every_depth]

            for edge in node_edges:
                dst = edge[1]
                if not visited[dst]:
                    queue.append((dst, depth + 1))
                    visited[dst] = True
        return solution

    @classmethod
    def create_available_graph(cls, vn, pn, vn_link):
        temp_graph = PhysicalNetwork(pn)
        unavailable_egdes = []
        for e in temp_graph.edges:
            for e_attr in vn.edge_attrs:
                if not e_attr.check(vn, temp_graph, vn_link, e):
                    unavailable_egdes.append(e)
                    break
        temp_graph.remove_edges_from(unavailable_egdes)
        return temp_graph

    @classmethod
    def release(cls, vn, pn, solution):
        r"""Release occupied resources when a VN leaves PN."""
        if solution['result'] == False:
            pass
        else:
            for vid, pid in solution['node_slots'].items():
                for n_attr in vn.get_node_attrs('resource'):
                    n_attr.update(vn.nodes[vid], pn.nodes[pid], '+')
            for vn_link, path in solution['edge_paths'].items():
                for e_attr in vn.get_edge_attrs('resource'):
                    e_attr.update_path(vn.edges[vn_link], pn, path, method='+')
        return True

    @classmethod
    def find_shortest_paths(cls, network, source, target, method='all_shortest', k=1):
        assert method in ['first_shortest', 'k_shortest', 'all_shortest']
        if method == 'first_shortest':
            shortest_path = nx.dijkstra_path(network, source, target)
            return [shortest_path]
        elif method == 'k_shortest':
            return list(islice(nx.shortest_simple_paths(network, source, target), k))
        elif method == 'all_shortest':
            return list(nx.all_shortest_paths(network, source, target))

    @classmethod
    def find_k_shortest_paths(cls, network, source, target, k):
        try:
            return list(islice(nx.shortest_simple_paths(network, source, target), k))
        except:
            return []

    @classmethod
    def find_candidate_nodes(cls, pn, vn, vn_node_id, filter=None, check_edge_constraint=False):
        r"""Find candicate nodes according to the restrictions and filter.

        Returns:
            candicate_nodes (list)
        """
        candidate_nodes = list(pn.nodes)
        for attr in vn.get_node_attrs('resource'):
            attr_name = attr.name
            attr_req = vn.nodes[vn_node_id][attr_name]
            pn_attr_data = np.array(list(nx.get_node_attributes(pn, attr_name).values()))
            suitable_nodes = np.where(pn_attr_data >= attr_req)
            candidate_nodes = np.intersect1d(candidate_nodes, suitable_nodes)
        candidate_nodes = np.setdiff1d(candidate_nodes, filter)
        return candidate_nodes.tolist()

    @classmethod
    def link_mapping(cls, vn, pn, solution, ordered_vn_edges=None, available_network=False, shortest_method='all_shortest', k=1, inplace=True):
        """link Mapping

        Args:
            vn ([type]): [description]
            pn ([type]): [description]
            ordered_vn_edges ([type]): [description]
            solution ([type]): [description]
            method (str, optional): [description]. Defaults to 'available'.
            inplace (bool, optional): [description]. Defaults to True.

        Returns:
            [type]: [description]
        """
        pn = pn if inplace else copy.deepcopy(pn)
        ordered_vn_edges = ordered_vn_edges if ordered_vn_edges is not None else list(vn.edges)
        node_slots = solution['node_slots']

        for vn_link in ordered_vn_edges:
            route_result = Controller.route(vn, pn, vn_link, (node_slots[vn_link[0]], node_slots[vn_link[1]]), 
                                            solution, available_network=available_network, shortest_method=shortest_method, k=k)
            if not route_result:
                # FAILURE
                solution['route_result'] = False
                return False
        # SUCCESS
        assert len(solution['edge_paths']) == vn.num_edges
        return True

    @classmethod
    def node_mapping(cls, vn, pn, ordered_vn_nodes, ordered_pn_nodes, solution, reusable=False, inplace=True):
        pn = pn if inplace else copy.deepcopy(pn)
        ordered_pn_nodes = copy.deepcopy(ordered_pn_nodes)

        for vid in ordered_vn_nodes:
            for pid in ordered_pn_nodes:
                place_result = Controller.place(vn, pn, vid, pid, solution)
                if place_result:
                    if reusable == False: ordered_pn_nodes.remove(pid)
                    break
            
            if not place_result:
                # FAILURE
                solution['place_result'] = False
                return False
        # SUCCESS
        assert len(solution['node_slots']) == vn.num_nodes
        return True

    @staticmethod
    def path_to_edges(path):
        assert len(path) > 1
        return [(path[i], path[i+1]) for i in range(len(path)-1)]

    @classmethod
    def update_pn_with_node_slots(cls, vn, pn, node_slots):
        for vid, pid in node_slots.items():
            for n_attr in vn.get_node_attrs('resource'):
                n_attr.update(vn.nodes[vid], pn.nodes[pid], '+')
            return True

if __name__ == '__main__':
    pass
