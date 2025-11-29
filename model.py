# model.py
import torch.nn as nn
import torchvision.models as models
import torch
import torch.nn.functional as F
import math

class ImageClassifier(nn.Module):
    def __init__(self, num_classes, architecture='resnet', weights=None):
        """
        num_classes: Number of output classes.
        architecture: 'resnet' or 'mobilenet'.
        weights: Pre-trained weights. If None, defaults will be used.
        """
        super(ImageClassifier, self).__init__()

        if architecture.lower() == 'resnet':
            weights = weights or models.ResNet18_Weights.DEFAULT
            self.model = models.resnet18(weights=weights)
            num_features = self.model.fc.in_features
            self.model.fc = nn.Linear(num_features, num_classes)

        elif architecture.lower() == 'mobilenet':
            weights = weights or models.MobileNet_V3_Large_Weights.DEFAULT
            self.model = models.mobilenet_v3_large(weights=weights)
            num_features = self.model.classifier[-1].in_features
            self.model.classifier[-1] = nn.Linear(num_features, num_classes)

        else:
            raise ValueError("architecture must be 'resnet' or 'mobilenet'")

    def forward(self, x):
        return self.model(x)
    

class ImageClassifierWithMLP(nn.Module):
    def __init__(self, num_classes, backbone='resnet', mlp_hidden=256, alpha_sharpness = 10):
        super().__init__()
        if backbone == 'resnet':
            self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()  # remove final fc
        elif backbone == 'mobilenet':
            self.backbone = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
            in_features = self.backbone.classifier[-1].in_features
            self.backbone.classifier[-1] = nn.Identity()
        else:
            raise ValueError("Backbone not supported")
        
        # Hebbian MLP
        self.mlp = HebbianMLP(layer_sizes=[in_features, mlp_hidden, num_classes], gate_fn=self.gate_fn)
        self.alpha_sharpness = alpha_sharpness
        
    def forward(self, x, return_activations=False):
        features = self.backbone(x)
        return self.mlp(features, return_activations=return_activations)
    

    def gate_fn(self, module, x=None, y_true=None, y_pred=None):
        # Convert y_true to one-hot if provided
        if y_true is None or y_pred is None:
            err = torch.ones(module.W_prox.shape[0], device=module.W_prox.device, dtype=module.W_prox.dtype)
        else:
            num_classes = y_pred.shape[1]
            y_true_onehot = F.one_hot(y_true, num_classes=num_classes).float()
            # Compute per-output-unit error
            err = torch.mean(torch.abs(y_true_onehot - y_pred), dim=0)  # shape: (out_features,)

        # Per-neuron input activation norm
        act_norm = x.norm(dim=1)  # shape: (batch,)
        act_norm = act_norm.unsqueeze(1).expand(-1, module.W_prox.shape[0])  # shape: (batch, out_features)
        act_norm = act_norm.mean(dim=0)  # shape: (out_features,)

        # Gate scaling with sigmoid
        err_gate = torch.sigmoid(self.alpha_sharpness * err)
        act_gate = torch.sigmoid(self.alpha_sharpness * act_norm)

        return err_gate * act_gate  # shape: (out_features,)



class HebbianNeuron(nn.Module):
    """
    A single neuron with two compartments:
      - W_prox: proximal weights (learned by backprop, Parameter)
      - W_dist: distal weights (Hebbian-only, stored as buffer)
      - E: eligibility trace (buffer)
    Hebbian updates are performed via local_hebb_update (no_grad).
    Inspired by https://scitechdaily.com/groundbreaking-study-uncovers-the-brains-secret-rules-of-learning/
    """
    def __init__(
        self,
        in_features,
        out_features,
        alpha_hebb=1e-2,
        decay_dist=0.0,
        gate_fn=None,
        p_dropout=0.2,
        trace_decay=0.99
    ):
        super().__init__()
        # Proximal weights (standard trainable parameters)
        self.W_prox = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))

        # Distal weights (Hebbian) stored as a buffer so they are saved but not
        # treated as optimizer params.
        self.register_buffer('W_dist', torch.empty(out_features, in_features))

        # Eligibility trace (buffer)
        self.trace_decay = trace_decay
        self.register_buffer('E', torch.zeros(out_features, in_features))

        # Hebbian-related hyperparams
        self.alpha_hebb = alpha_hebb
        self.decay_dist = decay_dist
        self.gate_fn = gate_fn or (lambda module, **kw: 1.0)

        # Activation + dropout
        self.activation = nn.LeakyReLU(0.01)
        self.dropout = nn.Dropout(p_dropout)

        self.reset_parameters()

    def reset_parameters(self):
        # Initialize proximal params
        nn.init.kaiming_uniform_(self.W_prox, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.W_prox)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0.0
        nn.init.uniform_(self.bias, -bound, bound)
        nn.init.normal_(self.W_dist, mean=0.0, std=1e-3)   # or nn.init.zeros_(self.W_dist)        
        nn.init.zeros_(self.E)

    def forward(self, x, return_activations=False):
        """
        x: (batch, in_features)
        If return_activations is True, returns (out, (pre, post)) for Hebbian updates.
        """
        y_prox = F.linear(x, self.W_prox, self.bias)  
        y_dist = F.linear(x, self.W_dist, None)    
        y = y_prox + y_dist
        out = self.activation(y)
        out = self.dropout(out)

        if return_activations:
            return out, (x, out)
        return out

    @torch.no_grad()
    def local_hebb_update(self, pre_act, post_act, y_true=None, y_pred=None):
        """
        pre_act: (batch, in_features)
        post_act: (batch, out_features)
        gate_args: forwarded to gate_fn (e.g. y_true, y_pred, etc.)
        """
        batch_size = pre_act.shape[0] if pre_act.shape[0] > 0 else 1
        hebb = torch.einsum('bi,bj->ij', post_act, pre_act) / float(batch_size)

        # Update eligibility trace: decay + accumulate
        self.E.mul_(self.trace_decay)
        self.E.add_(hebb)

        # Weight decay on distal weights
        if self.decay_dist and self.decay_dist > 0:
            self.W_dist.mul_(1.0 - float(self.decay_dist))
            

        gate = self.gate_fn(self, x=pre_act, y_true=y_true, y_pred=y_pred)
        # ----- SAFE gate handling (no float(...) which can move tensors to CPU) -----
        # Normalize gate into a tensor on the correct device/dtype for in-place ops.
        if isinstance(gate, torch.Tensor):
            # ensure same device and dtype as weights/traces
            gate = gate.to(self.W_dist.device).to(self.W_dist.dtype)
            if gate.numel() == 1:
                # scalar tensor OK (broadcasts)
                pass
            else:
                # allow per-output gating: make shape (out_features, 1) to broadcast with E
                gate = gate.view(-1, 1)
        else:
            # python scalar (int/float) -> convert to tensor on correct device
            gate = torch.tensor(gate, dtype=self.W_dist.dtype, device=self.W_dist.device)

        # Apply Hebbian trace scaled by alpha and gate (works with scalar or per-neuron gate)
        self.W_dist.add_(gate * self.alpha_hebb * self.E)

        # Row-normalize to keep scale stable (per-output neuron)
        row_norm = self.W_dist.norm(dim=1, keepdim=True).clamp(min=1e-6)
        self.W_dist.div_(row_norm)



class HebbianMLP(nn.Module):
    def __init__(self, layer_sizes, alpha_hebb=1e-2, decay_dist=1e-4,
                 trace_decay=0.99, gate_fn=None, p_dropout=0.2, last_p_dropout=0.0, hebb_start = None):
        super().__init__()
        num_layers = len(layer_sizes) - 1
        layers = []
        for i, (in_f, out_f) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            # use p_dropout for all but the last; last uses last_p_dropout (default 0.0)
            p = p_dropout if i < (num_layers - 1) else last_p_dropout
            layers.append(HebbianNeuron(in_f, out_f,
                                        alpha_hebb=alpha_hebb,
                                        decay_dist=decay_dist,
                                        gate_fn=gate_fn,
                                        p_dropout=p,
                                        trace_decay=trace_decay))
        self.layers = nn.ModuleList(layers)
        self.hebb_start = hebb_start if hebb_start is not None else (len(layers)-1)  # default: last layer
        print(f"Hebb start layer: {self.hebb_start}")


    def forward(self, x, return_activations=False):
        activations = []
        out = x
        for layer in self.layers:
            if return_activations:
                out, acts = layer(out, return_activations=True)
                activations.append(acts)  # acts is (pre, post)
            else:
                out = layer(out)
        return (out, activations) if return_activations else (out, None)
    

    def apply_hebb(self, acts, y_true=None, y_pred=None, gate_threshold=0.2, log_gates=False):
        """
        Apply Hebbian updates using activations collected from a forward pass.
        Optionally logs per-layer gate histograms.
        """
        with torch.no_grad():
            for i, (layer, (pre, post)) in enumerate(zip(self.layers, acts)):
                if i < self.hebb_start:
                    continue

                gate = layer.gate_fn(layer, x=pre, y_true=y_true, y_pred=y_pred)
                mean_gate = gate.mean().item()

                if log_gates:
                    # histogram as CPU numpy for TensorBoard/print
                    hist = gate.detach().cpu().numpy()
                    print(f"[Hebb][Layer {i}] Gate mean={mean_gate:.4f}, "
                        f"min={hist.min():.4f}, max={hist.max():.4f}")
                    # Optional: send to TensorBoard
                    # self.tb_writer.add_histogram(f'hebb/gate_layer_{i}', hist, global_step)

                if mean_gate < gate_threshold:
                    continue

                layer.local_hebb_update(pre.detach(), post.detach(), y_true=y_true, y_pred=y_pred)


    def ewc_loss(self, fisher, params_old, lambda_ewc=1.0):
            loss = 0.0
            for name, p in self.named_parameters():
                if name in fisher:
                    loss = loss + (fisher[name] * (p - params_old[name]).pow(2)).sum()
            return 0.5 * float(lambda_ewc) * loss
