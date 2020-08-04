void StaticAllocate(Graph &graph) {
  for(auto &v : graph.values) {
    v.value = at::empty(…);
    v.grad = at::empty(…);
  }
  for(auto &g : graph.differentiable_subgraphs) {
    StaticAllocate(g.forward_graph);
    StaticAllocate(g.backward_graph);
  }
}

CUDAGraph fcg = GetForwardCUDAGraph(graph);
CUDAGraph bcg = GetBackwardCUDAGraph(graph);

CUDAGraph GetForwardCUDAGraph(Graph graph) {
  CUDAGraph ret;
  for(auto &n : graph.nodes) {
    switch (n.kind) {
    case operator:
      CUDAGraph cg = op_registry.find(n.signature).createCUDAGraph(n.inputs, n.outputs);
      ret.merge_with(cg);
      break;
    case differentiable_subgraph:
      ret.merge_with(GetForwardCUDAGraph(n.forward_graph));
      break;
    case fusion_group:
      ret.merge_with(GetFusionCUDAGraph(n));
      break;
    }
  }
}

CUDAGraph GetBackwardCUDAGraph(Graph graph) {
  CUDAGraph ret;
  for(auto &n : graph.nodes) {
    switch (n.kind) {
    case operator:
      CUDAGraph cg = op_registry.find(n.signature).createBackwardCUDAGraph(n.inputs, n.outputs);
      ret.merge_with(cg);
      break;
    case differentiable_subgraph:
      ret.merge_with(GetForwardCUDAGraph(n.backward_graph));
      break;
    }
  }
}