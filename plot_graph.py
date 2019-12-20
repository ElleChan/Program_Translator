import matplotlib.pyplot as plt


handle = open('loss_data.txt')
lines = handle.readlines()
handle.close()

y_data = []
for y in lines:
    y_data.append(float(y))
x_data = [i for i in range(1,len(y_data)+1)]

print("x points:", x_data)
print("y points:", y_data)

plt.plot(x_data, y_data)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.savefig('loss_graph.png')
