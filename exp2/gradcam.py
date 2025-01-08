import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as T

class GradCAM:
    def __init__(self, model):
        self.model = model
        self.device = next(model.parameters()).device
        self.gradients = None
        self.activations = None

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        for name, module in self.model.named_modules():
            if name == self.target_layer:
                module.register_forward_hook(forward_hook)
                module.register_backward_hook(backward_hook)

    def generate_cam(self, input_tensor, target_layer, target_class=None):
        self.target_layer = target_layer
        self._register_hooks()

        self.model.eval()
        outputs = self.model(input_tensor)

        if target_class is None:
            target_class = outputs.argmax(dim=1).item()

        self.model.zero_grad()
        loss = outputs[:, target_class]
        loss.backward()

        gradients = self.gradients.cpu().data.numpy()
        activations = self.activations.cpu().data.numpy()
        weights = np.mean(gradients, axis=(2, 3))

        cam = np.zeros(activations.shape[2:], dtype=np.float32)
        for i, w in enumerate(weights[0]):
            cam += w * activations[0, i, :, :]

        cam = np.maximum(cam, 0)
        cam = cam - np.min(cam)
        cam = cam / np.max(cam) if np.max(cam) != 0 else cam

        return cam

    def visualize(self, img, target_layer, alpha=0.5, target_class=None):
        img = img.unsqueeze(0)
        img = img.to(self.device)
        cam = self.generate_cam(img, target_layer, target_class)
        img = img.squeeze().cpu().numpy()
        
        cam = np.interp(cam, (cam.min(), cam.max()), (0, 1))
        cam = np.uint8(cam * 255)

        cmap = plt.get_cmap('magma')
        heatmap = cmap(cam / 255.0)[:, :, :3]

        heatmap = T.ToPILImage()(heatmap)
        heatmap = heatmap.resize((img.shape[1], img.shape[0]))
        heatmap = T.ToTensor()(heatmap).permute(1, 2, 0).numpy()

        superimposed_img = (1 - alpha) * np.stack([img]*3, axis=-1) + alpha * heatmap

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 3, 1)
        plt.title("Original Image")
        plt.imshow(img, cmap='gray')
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.title("Heatmap")
        plt.imshow(heatmap)
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.title("Superimposed")
        plt.imshow(superimposed_img)
        plt.axis("off")

        plt.show()
