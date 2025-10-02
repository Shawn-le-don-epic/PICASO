from PIL import Image
image = Image.open('OPimg1.png')
print(f"Current size : {image.size}")
resized_image = image.resize((1500, 2249))
resized_image.save('OPimg1-resized.png')