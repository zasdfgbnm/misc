# to get the graph
input1 = torch.randn(...)
input2 = torch.randn(...)

with torch.cuda.capture_graph() as graph:
    output1, output2 = usercode(input1, input2)

# to use the graph
input1.copy_(real_input1)
input2.copy_(real_input2)
graph.run()
print(output1, output2)