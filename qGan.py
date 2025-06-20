import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
import matplotlib.pyplot as plt
from tqdm import tqdm

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

## Quantum Circuit Setup
num_qubits = 4
num_layers = 3
dev = qml.device("default.qubit", wires=num_qubits)

## 1. Quantum Generator
class QuantumGenerator(nn.Module):
    def __init__(self, num_qubits, num_layers):
        super().__init__()
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        # Initialize parameters (3 rotations per qubit per layer)
        self.params = nn.Parameter(torch.rand(num_layers, num_qubits, 3) * 2 * np.pi, dtype = np.complex128)
        
        # Define quantum circuit template
        @qml.qnode(dev, interface="torch")
        def circuit(params):
            # Start from |0> state
            for layer in range(num_layers):
                # Apply rotations to each qubit
                for qubit in range(num_qubits):
                    qml.Rot(params[layer, qubit, 0], 
                           params[layer, qubit, 1], 
                           params[layer, qubit, 2], 
                           wires=qubit)
                
                # Add entangling gates (linear entanglement)
                for qubit in range(num_qubits - 1):
                    qml.CNOT(wires=[qubit, qubit + 1])
            
            return qml.state()
        
        self.circuit = circuit
        # print(qml.draw(self.circuit)(self.params))
    
    def forward(self):
        """Generate a quantum state"""
        return self.circuit(self.params)

## 2. Quantum Discriminator
class QuantumDiscriminator(nn.Module):
    def __init__(self, num_qubits, num_layers):
        super().__init__()
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        # Initialize parameters (3 rotations per qubit per layer)
        self.params = nn.Parameter(torch.rand(num_layers, num_qubits, 3) * 2 * np.pi)
        
        # Define quantum circuit template
        @qml.qnode(dev, interface="torch")
        def circuit(state, params):
            # Prepare input state using StatePrep
            qml.StatePrep(state, wires=range(num_qubits))
            
            # Apply variational circuit
            for layer in range(num_layers):
                # Apply rotations to each qubit
                for qubit in range(num_qubits):
                    qml.Rot(params[layer, qubit, 0], 
                           params[layer, qubit, 1], 
                           params[layer, qubit, 2], 
                           wires=qubit)
                
                # Add entangling gates
                for qubit in range(num_qubits - 1):
                    qml.CNOT(wires=[qubit, qubit + 1])
            
            # Measure the first qubit (probability of |1> state)
            return qml.probs(wires=0)
        
        self.circuit = circuit


        # My code
        # real_part = torch.randn(2**num_qubits)
        # imag_part = torch.randn(2**num_qubits)
        # state = torch.complex(real_part, imag_part)
        
        # # Normalize to unit length
        # state = state / torch.norm(state)

        # print("\n\n\n")
        # print(qml.draw(self.circuit)(state, self.params))
    
    def forward(self, state):
        """Discriminate quantum state - returns probability of being fake"""
        # Circuit returns [P(0), P(1)] for the first qubit
        probs = self.circuit(state, self.params)
        # Use probability of |1> as "fake" score
        return probs[1].unsqueeze(0)

## 3. Quantum GAN
class QuantumGAN(nn.Module):
    def __init__(self, num_qubits=2, num_layers=3):
        super().__init__()
        self.generator = QuantumGenerator(num_qubits, num_layers)
        self.discriminator = QuantumDiscriminator(num_qubits, num_layers)
    
    def generate_real_samples(self, batch_size):
        """Generate batch of random pure quantum states"""
        real_states = []
        for _ in range(batch_size):
            # Create random complex vector
            real_part = torch.randn(2**num_qubits)
            imag_part = torch.randn(2**num_qubits)
            state = torch.complex(real_part, imag_part)
            
            # Normalize to unit length
            state = state / torch.norm(state)
            real_states.append(state)
        return real_states
    
    def train(self, epochs=200, batch_size=10, lr=0.05):
        opt_gen = optim.Adam(self.generator.parameters(), lr=lr)
        opt_disc = optim.Adam(self.discriminator.parameters(), lr=lr)
        
        # Store losses for plotting
        disc_losses = []
        gen_losses = []
        
        for epoch in tqdm(range(epochs)):
            # 1. Train Discriminator
            real_states = self.generate_real_samples(batch_size)
            fake_states = [self.generator().detach() for _ in range(batch_size)]
            
            # Reset gradients
            opt_disc.zero_grad()
            
            # Calculate discriminator loss
            disc_loss = 0
            for real_state in real_states:
                # Discriminator should output low for real states
                output = self.discriminator(real_state)
                disc_loss += torch.log(1 - output + 1e-8)
            
            for fake_state in fake_states:
                # Discriminator should output high for fake states
                output = self.discriminator(fake_state)
                disc_loss += torch.log(output + 1e-8)
            
            disc_loss = -disc_loss / (2 * batch_size)  # Negative log likelihood
            
            # Backpropagate and update
            disc_loss.backward()
            opt_disc.step()
            
            # 2. Train Generator
            opt_gen.zero_grad()
            
            # Generate new fake states
            fake_states = [self.generator() for _ in range(batch_size)]
            
            # Calculate generator loss
            gen_loss = 0
            for fake_state in fake_states:
                # Generator wants discriminator to think fakes are real
                output = self.discriminator(fake_state)
                gen_loss += torch.log(1 - output + 1e-8)
            
            gen_loss = -gen_loss / batch_size
            
            # Backpropagate and update
            gen_loss.backward()
            opt_gen.step()
            
            # Store losses
            disc_losses.append(disc_loss.item())
            gen_losses.append(gen_loss.item())
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch:3d} | D_loss: {disc_loss.item():.4f} | G_loss: {gen_loss.item():.4f}")
        
        # Plot training history
        # plt.figure(figsize=(10, 5))
        # plt.plot(disc_losses, label="Discriminator Loss")
        # plt.plot(gen_losses, label="Generator Loss")
        # plt.xlabel("Epoch")
        # plt.ylabel("Loss")
        # plt.title("Training Progress")
        # plt.legend()
        # plt.grid(True)
        # plt.savefig("qgan_training.png")
        # plt.show()

## 4. Evaluation Metrics
def fidelity(state1, state2):
    """Calculate quantum state fidelity"""
    return torch.abs(torch.dot(torch.conj(state1), state2))**2

def evaluate_model(qgan, num_tests=10):
    """Evaluate the trained model"""
    print("\n" + "="*50)
    print("Model Evaluation")
    print("="*50)
    
    # Test discriminator performance
    real_states = qgan.generate_real_samples(num_tests)
    fake_states = [qgan.generator().detach() for _ in range(num_tests)]
    
    real_scores = []
    fake_scores = []
    for state in real_states:
        real_scores.append(qgan.discriminator(state).item())
    for state in fake_states:
        fake_scores.append(qgan.discriminator(state).item())
    
    print(f"Discriminator on real states: Avg = {np.mean(real_scores):.4f}, Std = {np.std(real_scores):.4f}")
    print(f"Discriminator on fake states: Avg = {np.mean(fake_scores):.4f}, Std = {np.std(fake_scores):.4f}")
    
    # Test state generation quality
    fidelities = []
    for _ in range(num_tests):
        real_state = qgan.generate_real_samples(1)[0].to(torch.complex128)
        fake_state = qgan.generator().detach()
        fidelities.append(fidelity(real_state, fake_state).item())
    
    print(f"Fidelity between real and generated states: Avg = {np.mean(fidelities):.4f}, Std = {np.std(fidelities):.4f}")
    
    # Visualize a generated state
    real_state = qgan.generate_real_samples(1)[0].detach().numpy()
    fake_state = qgan.generator().detach().numpy()
    
    # plt.figure(figsize=(12, 5))
    
    # plt.subplot(1, 2, 1)
    # plt.bar(range(len(real_state)), np.abs(real_state))
    # plt.title("Real State (Magnitude)")
    # plt.xlabel("State Index")
    # plt.ylabel("Amplitude")
    
    # plt.subplot(1, 2, 2)
    # plt.bar(range(len(fake_state)), np.abs(fake_state))
    # plt.title("Generated State (Magnitude)")
    # plt.xlabel("State Index")
    # plt.ylabel("Amplitude")
    
    # plt.tight_layout()
    # plt.savefig("qgan_states.png")
    # plt.show()

## 5. Run the Quantum GAN
if __name__ == "__main__":
    print("Initializing Quantum GAN...")
    qgan = QuantumGAN(num_qubits=num_qubits, num_layers=num_layers)
    
    print("Starting training...")
    qgan.train(epochs=200, batch_size=10, lr=4e-3) #200 before
    
    print("Evaluating model...")
    evaluate_model(qgan)