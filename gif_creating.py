import networkx as nx
import matplotlib.pyplot as plt
import imageio

nodes = ["Market Data", "Feature Engine", "Regime Detector", 
         "Planner", "RL Agent", "Risk Manager", "Memory"]
edges = [("Market Data", "Feature Engine"),
         ("Feature Engine", "Regime Detector"),
         ("Regime Detector", "Planner"),
         ("Planner", "RL Agent"),
         ("RL Agent", "Risk Manager"),
         ("Risk Manager", "Memory")]

G = nx.DiGraph()
G.add_nodes_from(nodes)
G.add_edges_from(edges)

frames = []
for i in range(len(edges)):
    plt.figure(figsize=(8,5))
    colors = ['red' if j<=i else 'lightblue' for j in range(len(nodes))]
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_color=colors, node_size=2000, arrowsize=20)
    filename = f"frame_{i}.png"
    plt.savefig(filename)
    plt.close()
    frames.append(filename)

# Save GIF
images = [imageio.imread(f) for f in frames]
imageio.mimsave('architecture.gif', images, duration=1)
