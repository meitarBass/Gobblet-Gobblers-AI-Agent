import torch
import torch.nn as nn
import torch.nn.functional as F


class FiLMLayer(nn.Module):
    def __init__(self, condition_size, feature_size):
        super(FiLMLayer, self).__init__()
        # FiLM generates gamma and beta for feature-wise modulation
        self.gamma_net = nn.Linear(condition_size, feature_size)
        self.beta_net = nn.Linear(condition_size, feature_size)

    '''
    *****************************************************************************
    * x: (batch_size, feature_size)
    * condition: (batch_size, condition_size) - one-hot encoded vector
    * Apply feature-wise linear modulation: gamma * x + beta
    ***************************************************************************** 
    '''

    def forward(self, x, condition):
        condition = condition.float()
        gamma = self.gamma_net(condition)  # (batch_size, feature_size)
        beta = self.beta_net(condition)  # (batch_size, feature_size)
        return gamma * x + beta


class FeatureNet(nn.Module):

    # *****************************************************************************
    # Total of 22 features
    # Hidden layer : 120 nodes, each connected to 2 features
    # *****************************************************************************

    def __init__(self):
        super(FeatureNet, self).__init__()
        self.input_size = 22
        self.hidden_size = 120
        self.output_size = 1

        self.group1_indices = list(range(0, 6)) + list(range(18, 22))  # location features
        self.group2_indices = list(range(6, 18))  # number features

        # Precompute all possible pairs
        self.pairs = []
        for i in self.group1_indices:
            for j in self.group2_indices:
                self.pairs.append((i, j))

        # Each hidden node has 2 weights and a bias
        self.hidden_weights = nn.Parameter(torch.randn(self.hidden_size, 2))
        self.hidden_bias = nn.Parameter(torch.zeros(self.hidden_size))

        # FiLM layer for modulating hidden layer with one-hot encoded condition
        self.film = FiLMLayer(condition_size=3, feature_size=self.hidden_size)

        # Output layer (for now, just a placeholder)
        self.output = nn.Linear(self.hidden_size, self.output_size)

    # *****************************************************************************
    # x: (batch_size, 22)
    # Split input into features and condition
    # *****************************************************************************
    def forward(self, x):
        features = x[:, :22]
        condition = x[:, 22:]

        # Build the hidden layer activations
        h = []
        for idx, (i, j) in enumerate(self.pairs):
            xi = features[:, i]
            xj = features[:, j]
            hi = self.hidden_weights[idx, 0] * xi + self.hidden_weights[idx, 1] * xj + self.hidden_bias[idx]
            h.append(hi)
        h = torch.stack(h, dim=1)
        h = self.film(h, condition)  # Apply FiLM modulation
        h = F.relu(h)  # Apply activation func

        out = self.output(h)
        return out


def train_feature_net(model, train_loader, num_epochs=100, learning_rate=0.001, device='cpu'):
    model = model.to(device)
    model.train()

    # Define loss function and optimizer
    criterion = nn.MSELoss()  # Mean Squared Error for regression
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print(f"Starting training for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        total_loss = 0.0
        num_batches = 0

        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs.squeeze(), batch_labels)

            # Backward pass (automatic in PyTorch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / num_batches
            print(f"Epoch [{epoch + 1}/{num_epochs}], Average Loss: {avg_loss:.6f}")

    print("Training completed!")
    return model


def evaluate_feature_net(model, test_loader, device='cpu'):
    model = model.to(device)
    model.eval()

    criterion = nn.MSELoss()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch_features, batch_labels in test_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)

            outputs = model(batch_features)
            loss = criterion(outputs.squeeze(), batch_labels)

            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches
    print(f"Test Loss: {avg_loss:.6f}")
    return avg_loss


def create_data_loader(features, labels, batch_size=32, shuffle=True):
    from torch.utils.data import TensorDataset, DataLoader

    # Convert to tensors
    features_tensor = torch.tensor(features, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.float32)

    # Create dataset and dataloader
    dataset = TensorDataset(features_tensor, labels_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return dataloader


# Add to Feature_net.py
def extract_stage_heuristics(model):
    heuristics = {}
    for stage_idx, stage_name in enumerate(["early", "mid", "late"]):
        # Create condition vector for this stage
        condition = torch.zeros(1, 3)
        condition[0, stage_idx] = 1

        # Get FiLM parameters
        gamma = model.film.gamma_net(condition).detach().squeeze()
        beta = model.film.beta_net(condition).detach().squeeze()

        # Store with pair information
        stage_heuristics = []
        for idx, (i, j) in enumerate(model.pairs):
            stage_heuristics.append({
                "feature_pair": (i, j),
                "gamma": gamma[idx].item(),
                "beta": beta[idx].item(),
                "weights": model.hidden_weights[idx].detach().tolist()
            })

        heuristics[stage_name] = stage_heuristics
    return heuristics


# Map feature indices to human-readable names
FEATURE_MAP = {
    0: "Col1", 1: "Col2", 2: "Col3",
    3: "Row1", 4: "Row2", 5: "Row3",
    6: "P_S_on", 7: "P_M_on", 8: "P_L_on",
    9: "P_S_cap", 10: "P_M_cap", 11: "P_L_cap",
    12: "E_S_on", 13: "E_M_on", 14: "E_L_on",
    15: "E_S_cap", 16: "E_M_cap", 17: "E_L_cap",
    18: "Avg_X", 19: "Avg_Y", 20: "Spread",
    21: "Total_Pieces"
}


def interpret_heuristics(heuristics):
    for stage, data in heuristics.items():
        print(f"\n==== {stage.upper()} GAME HEURISTICS ====")
        sorted_data = sorted(data, key=lambda x: abs(x["gamma"]), reverse=True)[:10]  # Top 10

        for item in sorted_data:
            feat1, feat2 = item["feature_pair"]
            w1, w2 = item["weights"]
            print(f"• {FEATURE_MAP[feat1]} ({w1:.2f}) and {FEATURE_MAP[feat2]} ({w2:.2f})")
            print(f"  Gamma: {item['gamma']:.4f}, Beta: {item['beta']:.4f}")
            # Add human-readable interpretation
            if item["gamma"] > 0.5:
                print("  STRATEGY: Prioritize this feature combination")
            elif item["gamma"] < -0.5:
                print("  WARNING: Avoid this feature combination")


def generate_advice(feature_vector, heuristics, stage):
    advice = []
    feature_scores = {i: val for i, val in enumerate(feature_vector)}

    # Get top 3 heuristics for this stage
    stage_heuristics = sorted(heuristics[stage],
                              key=lambda x: abs(x["gamma"]),
                              reverse=True)[:3]

    for h in stage_heuristics:
        i, j = h["feature_pair"]
        current_score = feature_scores[i] * h["weights"][0] + feature_scores[j] * h["weights"][1]

        if h["gamma"] > 0 and current_score < 0.5:
            advice.append(f"Increase {FEATURE_MAP[i]} and {FEATURE_MAP[j]} interaction")
        elif h["gamma"] < 0 and current_score > 0.5:
            advice.append(f"Reduce {FEATURE_MAP[i]} and {FEATURE_MAP[j]} exposure")

    return advice


if __name__ == "__main__":
    # Example usage
    net = FeatureNet()
    dummy_input = torch.randn(4, 22)  # batch of 4 samples
    output = net(dummy_input)
    print("Output shape:", output.shape)

    # Example training setup
    print("\nExample training setup:")
    print("1. Prepare your features and labels")
    print("2. Create DataLoader: loader = create_data_loader(features, labels)")
    print("3. Train the model: train_feature_net(net, loader)")
    print("4. Evaluate: evaluate_feature_net(net, test_loader)")
