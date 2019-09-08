//
// Created by Baoxu Shi on 6/13/15.
//

#ifndef GBPEDIA_GRAPH_H
#define GBPEDIA_GRAPH_H

#include "node_loader.h"
#include "edge_loader.h"
#include "type_loader.h"
#include "edge_list.h"

#include <set>
#include <tuple>
#include <algorithm>    // std::max
#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <algorithm>
template<typename ND, typename TD>
class graph
{

  typedef ND node_type;
  typedef TD edge_type;

  std::shared_ptr<node_loader<node_type> > nodes_ptr;
  std::shared_ptr<edge_loader> edges_ptr;
  std::shared_ptr<type_loader<edge_type> > edgetypes_ptr;

  inline void is_node_valid(unsigned int id)
  {
    if (!nodes_ptr->exists(id)) {
      throw std::runtime_error("Nodes " + std::to_string(id) + " does not exist.");
    }
  }

  inline bool is_loop(std::vector<std::pair<unsigned int, unsigned int> > &path, unsigned int id) noexcept {
    for (auto it = path.cbegin(); it != path.cend(); ++it)
    {
      if (it->first == id) return true;
    }
    return false;
  }


  /**
   * Homogeneous dfs helper
   */
  void dfs_helper(unsigned int src, unsigned int dst,  std::vector<unsigned int> discard_rel, unsigned max_depth,
                  std::vector<unsigned int> &tmp_path, std::set<unsigned int> &visited,
                  std::vector<std::vector<unsigned int> > &result, bool is_directed, bool depth_only, unsigned depth)
  {
    if (tmp_path.size() > 0 && tmp_path.size() <= max_depth && tmp_path.back() == dst) {
      if (!depth_only || (depth_only && tmp_path.size() == max_depth)) {
        result.push_back(tmp_path);
      }
      return;
    }

    if (max_depth == depth) return;

    bool dst_visited = false; // on typed network, there may be multiple edges
    // connects to the same target with different edge types,
    // we need to ignore these in an untyped setting. This can be done using visited_set
    // but it can not work with the target node.

    edge_list &edges = edges_ptr->get_edges(src);
    for (auto it = edges.get_forward().cbegin();
         it != edges.get_forward().cend(); ++it) {


      if (std::find(discard_rel.begin(), discard_rel.end(), it->second) != discard_rel.end()) { 
        continue; 
      }

      if (visited.find(it->first) == visited.end() || (it->first == dst && !dst_visited)) { // never visited
        tmp_path.push_back(it->first);
        dfs_helper(it->first, dst, discard_rel, max_depth, tmp_path, visited, result, is_directed, depth_only,
                   depth + 1);
        tmp_path.pop_back();
        visited.insert(it->first);
        if (it->first == dst) {
          dst_visited = true;
        }
      }
    }

    if (!is_directed) {
      for (auto it = edges.get_backward().cbegin();
           it != edges.get_backward().cend(); ++it) {

      
        if (std::find(discard_rel.begin(), discard_rel.end(), it->second) != discard_rel.end()) { 
          continue; 
        }

        if (visited.find(it->first) == visited.end() || (it->first == dst && !dst_visited)) {
          tmp_path.push_back(it->first);
          dfs_helper(it->first, dst, discard_rel, max_depth, tmp_path, visited, result, is_directed, depth_only,
                     depth + 1);
          tmp_path.pop_back();
          visited.insert(it->first);
        }
      }
    }

  }


  /**
   * Heterogeneous dfs helper
   */
    // void dfs_helper(unsigned int src, unsigned int dst, std::vector<std::tuple<unsigned int, unsigned int, unsigned int>> discard_rel, unsigned max_depth,
    //               std::vector<std::pair<unsigned int, unsigned int> > &tmp_path, std::vector<bool> &reverted_rel,
    //               std::set<unsigned int> &visited,
    //               std::vector<std::vector<std::pair<unsigned int, unsigned int> > > &result,
    //               std::vector<std::vector<bool> > &rel_result, bool is_directed, unsigned depth)
  void dfs_helper(unsigned int src, unsigned int dst,  std::vector<unsigned int> discard_rel, unsigned max_depth,
                  std::vector<std::pair<unsigned int, unsigned int> > &tmp_path, std::vector<bool> &reverted_rel,
                  std::set<unsigned int> &visited,
                  std::vector<std::vector<std::pair<unsigned int, unsigned int> > > &result,
                  std::vector<std::vector<bool> > &rel_result, bool is_directed, unsigned depth)
  {
    if (tmp_path.size() > 0 && tmp_path.size() <= max_depth && tmp_path.back().first == dst) {
      result.push_back(tmp_path);
      rel_result.push_back(reverted_rel);
      return;
    }

    if (max_depth == depth) return;

    edge_list &edges = edges_ptr->get_edges(src);
    for (auto it = edges.get_forward().cbegin();
          it != edges.get_forward().cend(); ++it) {
  
      if(std::find(discard_rel.begin(), discard_rel.end(), it->second) != discard_rel.end()) {
        continue;
       }
           
      if (visited.find(it->first) == visited.end()) {
        tmp_path.push_back(*it);
        reverted_rel.push_back(false);
        visited.insert(it->first);
        dfs_helper(it->first, dst, discard_rel, max_depth, tmp_path, reverted_rel, visited, result, rel_result,
                   is_directed,
                   depth + 1);
        tmp_path.pop_back();
        reverted_rel.pop_back();
        visited.erase(it->first);
      }
    }
    
    if (!is_directed) {
      for (auto it = edges.get_backward().cbegin();
           it != edges.get_backward().cend(); ++it) {
      
        if(std::find(discard_rel.begin(), discard_rel.end(), it->second) != discard_rel.end()) {
           continue;
        }
         
      
        if (visited.find(it->first) == visited.end()) {
          tmp_path.push_back(*it);
          reverted_rel.push_back(true);
          visited.insert(it->first);
          dfs_helper(it->first, dst, discard_rel, max_depth, tmp_path, reverted_rel, visited, result, rel_result,
                     is_directed,
                     depth + 1);
          tmp_path.pop_back();
          reverted_rel.pop_back();
          visited.erase(it->first);
        }
      }
    }

  }

public:

  graph() : nodes_ptr(nullptr), edges_ptr(nullptr), edgetypes_ptr(nullptr) { };

  graph(node_loader<node_type> &nodes, edge_loader &edges, type_loader<edge_type> &edgetypes) :
    nodes_ptr(&nodes),
    edges_ptr(&edges),
    edgetypes_ptr(&edgetypes) { };

  const std::set<unsigned int> &get_out_edges(unsigned src)
  {
    is_node_valid(src);
    return edges_ptr->get_edges(src).get_out_neighbors();

  }

  const std::set<unsigned int> &get_in_edges(unsigned src)
  {
    is_node_valid(src);
    return edges_ptr->get_edges(src).get_in_neighbors();
  }

  size_t get_deg(unsigned src)
  {
    is_node_valid(src);
    return edges_ptr->get_edges(src).get_deg();
  }


  std::vector<std::vector<unsigned int> > homogeneous_dfs(unsigned int src, unsigned int dst, unsigned int discard_rel,
      unsigned int depth, bool depth_only, bool is_directed)
  {
    is_node_valid(src);
    is_node_valid(dst);

    
    std::vector<unsigned int> exclude_edges;

    exclude_edges.push_back(discard_rel);
    exclude_edges.push_back(edges_ptr->get_type_rel());

    std::vector<std::vector<unsigned int> > result;

    try {
      std::vector<unsigned int> tmp_path;
      std::set<unsigned int> visited;
      tmp_path.push_back(src);
      visited.insert(src);
      dfs_helper(src, dst, exclude_edges, depth, tmp_path, visited, result, is_directed, depth_only, 1u);
      assert(tmp_path.size() == 1); // only source node is in it.
    } catch (std::exception error) {
      std::cerr << "Error occurred when performing dfs, " << error.what() << std::endl;
    }

    return result;

  };

  std::pair<std::vector<std::vector<std::pair<unsigned int, unsigned int> > >, std::vector<std::vector<bool> > > heterogeneous_dfs(
    unsigned int src, unsigned int dst, unsigned int discard_rel, bool is_directed, unsigned int depth)
  {
    is_node_valid(src);
    is_node_valid(dst);

    
    std::vector<unsigned int> exclude_edges;

    exclude_edges.push_back(discard_rel);
    exclude_edges.push_back(edges_ptr->get_type_rel());

    std::vector<std::vector<std::pair<unsigned int, unsigned int> > > path_result;
    std::vector<std::vector<bool> > rel_result;

    try {
      std::vector<std::pair<unsigned int, unsigned int> > tmp_path;
      std::vector<bool> tmp_reverted;
      std::set<unsigned int> visited;
      visited.insert(src);
      dfs_helper(src, dst, exclude_edges, depth, tmp_path, tmp_reverted, visited, path_result, rel_result, is_directed,
                 1u);
      assert(tmp_path.size() == 0);
    } catch (std::exception error) {
      std::cerr << "Error occurred when performing heterogeneous dfs, " << error.what() << std::endl;
    }

    assert(path_result.size() == rel_result.size());

    return std::pair<std::vector<std::vector<std::pair<unsigned int, unsigned int> > >, std::vector<std::vector<bool> > >(
             path_result, rel_result);
  }

  node_type get_node_type(unsigned int id)
  {
    return nodes_ptr->get_value(id);
  }

  edge_type get_edge_type(unsigned int id)
  {
    return edgetypes_ptr->get_value(id);
  }


  unsigned int get_node_id(node_type value)
  {
    return nodes_ptr->get_index(value);    
  }

  unsigned int get_edge_id(edge_type value)
  {
    return edgetypes_ptr->get_index(value);    
  }



  inline std::vector<unsigned int> get_ontology(unsigned int id)
  {
    is_node_valid(id);
    return edges_ptr->get_ontology(id);
  }

  inline std::vector<std::pair<unsigned int, std::set<unsigned int> > > get_ontology_siblings(unsigned int id)
  {
    is_node_valid(id);
    return edges_ptr->get_ontology_siblings(id);
  }

  inline std::set<unsigned int> get_ontology_siblings(unsigned int id, double tol)
  {
    is_node_valid(id);
    return edges_ptr->get_ontology_siblings(id, tol);
  }

  inline std::vector<std::pair<unsigned int, unsigned int> > get_ontology_sibling_count(unsigned int id)
  {
    is_node_valid(id);
    return edges_ptr->get_ontology_sibling_count(id);
  }

  bool connected_by_helper(unsigned int src, unsigned int dst, unsigned int pos, std::vector<unsigned int> &link_type,
                           bool is_directed)
  {
    edge_list &edges = edges_ptr->get_edges(src);

    for (auto it = edges.get_forward().cbegin(); it != edges.get_forward().cend(); ++it) {
      if (it->second == link_type[pos]) { // match link type
        if (pos == link_type.size() - 1) { // reach the end
          if (it->first == dst) {
            return true;
          }
        } else { // not reach the end
          if (connected_by_helper(it->first, dst, pos + 1, link_type, is_directed)) { // if find it
            return true;
          }
        }
      }
    }

    if (!is_directed) {
      for (auto it = edges.get_backward().cbegin(); it != edges.get_backward().cend(); ++it) {
        if (it->second == link_type[pos]) { // match link type
          if (pos == link_type.size() - 1) { // reach the end
            if (it->first == dst) {
              return true;
            }
          } else { // not reach the end
            if (connected_by_helper(it->first, dst, pos + 1, link_type, is_directed)) { // if find it
              return true;
            }
          }
        }
      }
    }

    return false;
  }

  bool connected_by(unsigned int src, unsigned int dst, std::vector<unsigned int> link_type, bool is_directed = false)
  {
    is_node_valid(src);
    is_node_valid(dst);

    return (connected_by_helper(src, dst, 0, link_type, is_directed));

  }

  bool connected_by(unsigned int src, unsigned int dst, unsigned int link_type, bool is_directed = false)
  {
    is_node_valid(src);
    is_node_valid(dst);

    edge_list &edges = edges_ptr->get_edges(src);

    for (auto it = edges.get_forward().cbegin(); it != edges.get_forward().cend(); ++it) {
      if (it->first == dst && it->second == link_type) return true;
    }

    if (!is_directed) {
      for (auto it = edges.get_backward().cbegin(); it != edges.get_backward().cend(); ++it) {
        if (it->first == src && it->second == link_type) return true;
      }
    }

    return false;
  }

  //TODO: This function iterates through entire edge list to find qualified entity pairs, try cache this
  std::vector<std::tuple<unsigned int, unsigned int, unsigned int, node_type, node_type> > get_entity_pairs_by_rel(edge_type rel_name,
      double sample_rate = 0.1)
  {

    srand(233);
    unsigned int rel_type = edgetypes_ptr->get_index(rel_name);
    std::vector<std::tuple<unsigned int, unsigned int, unsigned int, node_type, node_type> > res;
    for (unsigned int i = nodes_ptr->getMin_id(); i <= nodes_ptr->getMax_id(); i++) {
      edge_list &edges = edges_ptr->get_edges(i);
      for (auto it = edges.get_forward().cbegin(); it != edges.get_forward().cend(); ++it) {
        if (it->second == rel_type && ((double) rand() / RAND_MAX <= sample_rate)) {
          res.push_back(std::tuple<unsigned int, unsigned int, unsigned int, node_type, node_type>(i, it->first, rel_type, nodes_ptr->get_value(i), nodes_ptr->get_value(it->first)));
        }
      }
    }

    return res;
  }

 std::vector<std::pair<node_type, node_type> > get_overlapping_pairs(edge_type rel_1, edge_type rel_2, bool fb) {
    unsigned int rel_type_1 = edgetypes_ptr->get_index(rel_1);
    unsigned int rel_type_2 = edgetypes_ptr->get_index(rel_2);
    unsigned int count = 0;
    std::vector<std::pair<node_type, node_type> > res;
    for (unsigned int i = nodes_ptr->getMin_id(); i <= nodes_ptr->getMax_id(); i++) {
      std::vector<unsigned int> res_1;
      std::vector<unsigned int> res_2;
      edge_list &edges = edges_ptr->get_edges(i);
      for (auto it = edges.get_forward().cbegin(); it != edges.get_forward().cend(); ++it) {
        if (it->second == rel_type_1) {
          res_1.push_back(it->first);
        } 
      }

      if (fb) {
        for (auto it = edges.get_forward().cbegin(); it != edges.get_forward().cend(); ++it) {
          if (it->second == rel_type_2) {
            res_2.push_back(it->first);
          }
          // else {
          //   edge_list &edges_2 = edges_ptr->get_edges(it->first);
          //   for (auto it_2 = edges_2.get_forward().cbegin(); it_2 != edges_2.get_forward().cend(); ++it_2) {
          //       if (it_2->second == rel_type_2) {
          //         res_2.push_back(it_2->first);
          //       }
          //   }
          // } 
        }

        // for (auto it = edges.get_backward().cbegin(); it != edges.get_backward().cend(); ++it) {
        //     edge_list &edges_2 = edges_ptr->get_edges(it->first);
        //     for (auto it_2 = edges_2.get_forward().cbegin(); it_2 != edges_2.get_forward().cend(); ++it_2) {
        //       if (it_2->second == rel_type_2) {
        //          res_2.push_back(it_2->first);
        //       }
        //     }
        // } 
      }
      else {
        for (auto it = edges.get_backward().cbegin(); it != edges.get_backward().cend(); ++it) {
          if (it->second == rel_type_2) {
            res_2.push_back(it->first);
          } 
        }
      }


      for (unsigned it_1 =0; it_1 < res_1.size(); it_1++) {
        for (unsigned it_2=0; it_2<res_2.size(); it_2++) {
            if (res_1.at(it_1) == res_2.at(it_2)) {
              res.push_back(std::pair<node_type, node_type>(nodes_ptr->get_value(i), nodes_ptr->get_value(res_1.at(it_1))));
            }
        }
      }

    }
    return res;    

  }

  inline std::set<std::pair<unsigned int, unsigned int> > get_entity_pairs_by_triple(unsigned int src, unsigned int dst,
      unsigned int rel_type,
      double sample_rate = 0.1)
  {
    return get_entity_pairs_by_triple_helper(src, dst, rel_type, sample_rate, false);
  }

  inline std::set<std::pair<unsigned int, unsigned int> > get_entity_pairs_without_rel(unsigned int src,
      unsigned int dst,
      unsigned int rel_type,
      double sample_rate = 0.1)
  {
    return get_entity_pairs_by_triple_helper(src, dst, rel_type, sample_rate, true);
  }

  std::set<std::pair<unsigned int, unsigned int> > get_entity_pairs_by_triple_helper(unsigned int src, unsigned int dst,
      unsigned int rel_type,
      double sample_rate = 0.1,
      bool exclude_rel = false)
  {
    is_node_valid(src);
    is_node_valid(dst);
    srand(233);

    // Step 1: Get src_set and dst_set that matches src and dst's ontology

    std::set<unsigned int> src_set = get_ontology_siblings(src, 0.0);
    std::set<unsigned int> dst_set = get_ontology_siblings(dst, 0.0);

    // Step 2: Get true labeled node pairs from previous sets

    std::set<std::pair<unsigned int, unsigned int> > entity_pairs;
    std::set<std::pair<unsigned int, unsigned int> > entity_pairs_with_rel;

    for (auto it = src_set.cbegin(); it != src_set.cend(); ++it) {
      auto edges = edges_ptr->get_edges(*it).get_forward();
      for (auto p = edges.cbegin(); p != edges.cend(); ++p) {
        if (!exclude_rel) { // get all entity pairs with rel_type
          if (p->second == rel_type && dst_set.find(p->first) != dst_set.end()) { // rel match and dst is in the set
            entity_pairs_with_rel.insert(std::pair<unsigned int, unsigned int>(*it, p->first));
          }
        } else { // get all entity pairs without rel_type
          if (p->second != rel_type && dst_set.find(p->first) != dst_set.end()) {
            entity_pairs.insert(std::pair<unsigned int, unsigned int>(*it, p->first));
          }
        }
      }
    }

    for (auto it = dst_set.cbegin(); it != dst_set.cend(); ++it) {
      auto edges = edges_ptr->get_edges(*it).get_backward();
      for (auto p = edges.cbegin(); p != edges.cend(); ++p) {
        if (!exclude_rel) {
          if (p->second == rel_type && src_set.find(p->first) != src_set.end()) { // rel match and src is in the set
            entity_pairs_with_rel.insert(std::pair<unsigned int, unsigned int>(p->first, *it));
          }
        } else { // get all entity pairs that matching the given ontology but no rel_type
          if (p->second != rel_type && src_set.find(p->first) != src_set.end()) {
            entity_pairs.insert(std::pair<unsigned int, unsigned int>(p->first, *it));
          }
        }

      }
    }

    if (!exclude_rel) return entity_pairs_with_rel;

    for (auto it = entity_pairs_with_rel.cbegin(); it != entity_pairs_with_rel.cend(); ++it) {
      auto p = entity_pairs.find(*it);
      if (p != entity_pairs.end()) {
        entity_pairs.erase(p);
      }
    }

    return entity_pairs;

  }

  inline unsigned int get_edge_type_count(unsigned int rel_type)
  {
    return edges_ptr->get_edge_type_count(rel_type);
  }

  std::vector<std::pair<unsigned int, node_type>> get_neighbor_by_rel(node_type src_, edge_type rel_type_, bool is_directed)
  {
    unsigned int src = nodes_ptr->get_index(src_);
    unsigned int rel_type = edgetypes_ptr->get_index(rel_type_);
    is_node_valid(src);

    edge_list &edges = edges_ptr->get_edges(src);

    std::vector<std::pair<unsigned int, node_type>> res;
    if (is_directed) {
      for (auto it = edges.get_forward().cbegin(); it != edges.get_forward().cend(); ++it) {
        if (it->second == rel_type) {
          res.push_back(std::pair<unsigned int, node_type>(it->first, nodes_ptr->get_value(it->first)));
        }
      }
    }

    else {
      for (auto it = edges.get_backward().cbegin(); it != edges.get_backward().cend(); ++it) {
        if (it->second == rel_type) {
          res.push_back(std::pair<unsigned int, node_type>(it->first, nodes_ptr->get_value(it->first)));
        }
      }  
    }
    return res;
  }

  std::vector<std::tuple<unsigned int, node_type, edge_type>> get_neighbors(unsigned int src, unsigned int dst, edge_type rel_type_, bool is_directed)
  {
    // unsigned int src = nodes_ptr->get_index(src_);
    unsigned int rel_type = edgetypes_ptr->get_index(rel_type_);
    is_node_valid(src);

    std::vector<std::tuple<unsigned int, node_type, edge_type>> res;

    edge_list &edges = edges_ptr->get_edges(src);

    std::vector<unsigned int> dst_ontology = get_ontology(dst);

    // std::set<unsigned int> neighbor_set = edges_ptr->get_neighbors(src, rel_type, is_directed);
    std::vector<unsigned int> src_ontology;
    unsigned int total = 0;
    if (is_directed) {
      for (auto it = edges.get_forward().cbegin(); it != edges.get_forward().cend(); ++it) {
        src_ontology = get_ontology(it->first);
        unsigned int count = 0;
        unsigned int p;
        unsigned int q;
        for (auto ii = dst_ontology.begin(); ii != dst_ontology.end(); ++ii) {
          for (auto jj = src_ontology.begin(); jj != src_ontology.end(); ++jj) {
            p = *ii;
            q = *jj;
            if (p == q) {
              count = count + 1;
            }         
          }
        }
        if ((rel_type == 0 || rel_type != it->second) && (count >= std::min(4,(int) dst_ontology.size()))) {
          res.push_back(std::tuple<unsigned int, node_type, edge_type>(it->first, nodes_ptr->get_value(it->first), edgetypes_ptr->get_value(it->second)));
          total = total + 1;
          if (total > 5) {
            break;
          }
        }
      }
    }
    if (!is_directed) {
      for (auto it = edges.get_backward().cbegin(); it != edges.get_backward().cend(); ++it) {
        src_ontology = get_ontology(it->first);
        unsigned int count = 0;
        unsigned int p;
        unsigned int q;
        for (auto ii = dst_ontology.begin(); ii != dst_ontology.end(); ++ii) {
          for (auto jj = src_ontology.begin(); jj != src_ontology.end(); ++jj) {
            p = *ii;
            q = *jj;
            if ( p == q) {
              count = count + 1;
            }         
          }
        }        
        if ((rel_type == 0 || rel_type != it->second) && (count >= std::min(4,(int) dst_ontology.size()))) {
          res.push_back(std::tuple<unsigned int, node_type, edge_type>(it->first, nodes_ptr->get_value(it->first), "(-1)"+edgetypes_ptr->get_value(it->second)));
          total = total + 1;
          if (total > 5) {
            break;
          }
        }
      }
    }

    return res;
  }


 /* 
  std::set<unsigned int> get_same_ontology(unsigned int id){
  return edges_ptr->get_same_ontology(id);
  }
 */

  unsigned int get_nontology()
  {
    return edges_ptr->get_nontology();
  }

};

#endif //GBPEDIA_GRAPH_H

