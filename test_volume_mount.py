import os
print("Saving file")
f = open("/tf/nn_checkpoints/demofile2.txt", "a")
f.write("Now the file has more content!")
f.close()
print(os.listdir())