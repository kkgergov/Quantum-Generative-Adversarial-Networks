import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pennylane as qml
import matplotlib.pyplot as plt
from tqdm import tqdm

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

## Quantum Circuit Setup
num_qubits = 2
num_layers = 3
dev = qml.device("default.qubit", wires=num_qubits)

## Quantum Circuit Templates
def generator_circuit(params, wires):
    """Quantum generator circuit"""
    num_qubits = len(wires)
    num_params = len(params) // (num_qubits * num_layers)
    params = params.reshape(num_layers, num_qubits, num_params)
    
    for layer in range(num_layers):
        # Apply rotations to all qubits
        for qubit in range(num_qubits):
            qml.Rot(*params[layer, qubit], wires=wires[qubit])
        
        # Add entangling gates (ring topology)
        for qubit in range(num_qubits):
            qml.CNOT(wires=[wires[qubit],wires[(qubit+1) % num_qubits]])
    
    return qml.state()

def discriminator_circuit(state, params, wires):
    """Quantum discriminator circuit"""
    num_qubits = len(wires)
    num_params = len(params) // (num_qubits * num_layers)
    params = params.reshape(num_layers, num_qubits, num_params)
    
    # Prepare input state
    qml.StatePrep(state, wires=wires)
    
    for layer in range(num_layers):
        # Apply rotations to all qubits
        for qubit in range(num_qubits):
            qml.Rot(*params[layer, qubit], wires=wires[qubit])
        
        # Add entangling gates (ring topology)
        for qubit in range(num_qubits):
            qml.CNOT(wires=[wires[qubit],wires[(qubit+1) % num_qubits]])
    
    # Measure the first qubit
    return qml.expval(qml.PauliZ(0))

## 1. Quantum Generator Module
class QuantumGenerator(nn.Module):
    def __init__(self, num_qubits, num_layers):
        super().__init__()
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        # Each layer has 3 parameters per qubit (RX, RY, RZ)
        self.params = nn.Parameter(
            torch.rand(num_layers * num_qubits * 3, dtype=torch.float64) * 2 * np.pi
        )
        
        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(params):
            return generator_circuit(params, wires=range(num_qubits))
        
        self.circuit = circuit
    
    def forward(self):
        """Generate a quantum state"""
        state = self.circuit(self.params)
        return state / torch.norm(state)  # Ensure normalization

## 2. Quantum Discriminator Module
class QuantumDiscriminator(nn.Module):
    def __init__(self, num_qubits, num_layers):
        super().__init__()
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        # Each layer has 3 parameters per qubit (RX, RY, RZ)
        self.params = nn.Parameter(
            torch.rand(num_layers * num_qubits * 3, dtype=torch.float64) * 2 * np.pi
        )
        
        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(state, params):
            return discriminator_circuit(state, params, wires=range(num_qubits))
        
        self.circuit = circuit
    
    def forward(self, state):
        """Discriminate quantum state - returns expectation value"""
        # Expectation value in [-1, 1]
        expectation = self.circuit(state, self.params)
        
        # Convert to probability in [0, 1] for BCE loss
        # P(fake) = (1 - expectation)/2
        return (1 - expectation) / 2

## 3. Quantum GAN
class QuantumGAN(nn.Module):
    def __init__(self, num_qubits=2, num_layers=3):
        super().__init__()
        self.generator = QuantumGenerator(num_qubits, num_layers)
        self.discriminator = QuantumDiscriminator(num_qubits, num_layers)
        self.target_state = self.create_target_state()
    
    def create_target_state(self):
        """Create a fixed target state to approximate"""
        real_part = np.random.randn(2**num_qubits).astype(np.float64)
        imag_part = np.random.randn(2**num_qubits).astype(np.float64)
        state = (real_part + 1j * imag_part).astype(np.complex128)
        state /= np.linalg.norm(state)
        return torch.tensor(state, dtype=torch.complex128)
    
    def generate_real_batch(self, batch_size):
        """Generate batch of real states (all same target state)"""
        return [self.target_state.clone() for _ in range(batch_size)]
    
    def train(self, epochs=200, batch_size=10, lr=0.01, patience=50, min_improvement=0.001):
        opt_gen = optim.Adam(self.generator.parameters(), lr=lr)
        opt_disc = optim.Adam(self.discriminator.parameters(), lr=lr)
        
        disc_losses = []
        gen_losses = []
        fidelities = []
        
        for epoch in tqdm(range(epochs)):

            # 1. Train Discriminator
            real_states = self.generate_real_batch(batch_size)
            fake_states = [self.generator().detach() for _ in range(batch_size)]
            
            opt_disc.zero_grad()
            disc_loss = 0
            
            # Process real states
            real_outputs = []
            for state in real_states:
                output = self.discriminator(state)
                real_outputs.append(output)
                disc_loss += F.binary_cross_entropy(output, torch.zeros_like(output))
            
            # Process fake states
            fake_outputs = []
            for state in fake_states:
                output = self.discriminator(state)
                fake_outputs.append(output)
                disc_loss += F.binary_cross_entropy(output, torch.ones_like(output))
            
            disc_loss = disc_loss / (2 * batch_size)
            disc_loss.backward()
            opt_disc.step()
            
            # 2. Train Generator
            opt_gen.zero_grad()
            gen_loss = 0
            fake_states = [self.generator() for _ in range(batch_size)]
            
            for state in fake_states:
                output = self.discriminator(state)
                gen_loss += F.binary_cross_entropy(output, torch.zeros_like(output))
            
            gen_loss = gen_loss / batch_size
            gen_loss.backward()
            opt_gen.step()
            
            # Store losses
            disc_losses.append(disc_loss.item())
            gen_losses.append(gen_loss.item())
            
            # Calculate fidelity with target state
            gen_state = self.generator().detach()
            fid = torch.abs(torch.vdot(self.target_state, gen_state))**2
            fidelities.append(fid.item())
            
            if epoch % 20 == 0:
                avg_real = torch.mean(torch.tensor(real_outputs)).item()
                avg_fake = torch.mean(torch.tensor(fake_outputs)).item()
                print(f"Epoch {epoch:3d} | D_loss: {disc_loss.item():.4f} | G_loss: {gen_loss.item():.4f}")
                print(f"  D(real): {avg_real:.4f} | D(fake): {avg_fake:.4f} | Fidelity: {fid.item():.4f}")
        
        # Plot results
        plt.figure(figsize=(15, 10))
        
        # Loss plot
        plt.subplot(2, 2, 1)
        plt.plot(disc_losses, label="Discriminator Loss")
        plt.plot(gen_losses, label="Generator Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.legend()
        plt.grid(True)
        
        # Fidelity plot
        plt.subplot(2, 2, 2)
        plt.plot(fidelities)
        plt.xlabel("Epoch")
        plt.ylabel("Fidelity")
        plt.title("State Fidelity with Target")
        plt.grid(True)
        
        # Final state visualization
        final_state = self.generator().detach().numpy()
        plt.subplot(2, 2, 3)
        plt.bar(range(len(self.target_state)), np.abs(self.target_state.numpy()))
        plt.title("Target State (Magnitude)")
        plt.xlabel("State Index")
        plt.ylabel("Amplitude")
        
        plt.subplot(2, 2, 4)
        plt.bar(range(len(final_state)), np.abs(final_state))
        plt.title("Generated State (Magnitude)")
        plt.xlabel("State Index")
        plt.ylabel("Amplitude")
        
        plt.tight_layout()
        plt.savefig("qgan_state_results.png")
        plt.show()
        
        return disc_losses, gen_losses, fidelities