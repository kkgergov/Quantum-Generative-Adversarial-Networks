import torch
import torch.nn as nn
import torch.optim as optim
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

## Helper functions for density matrices
def state_to_density(state):
    """Convert a state vector to a density matrix (pure state)"""
    state = state.reshape(-1, 1)  # Column vector
    return state @ state.conj().T

## Quantum Generator
class QuantumGenerator(nn.Module):
    def __init__(self, num_qubits, num_layers):
        super().__init__()
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.params = nn.Parameter(
            torch.rand(num_layers * num_qubits * 3, dtype=torch.float64) * 2 * np.pi
        )
        
        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(params):
            # Apply generator circuit
            for l in range(num_layers):
                # Single-qubit rotations
                for q in range(num_qubits):
                    idx = l*num_qubits*3 + q*3
                    qml.Rot(params[idx], params[idx+1], params[idx+2], wires=q)
                
                # Entanglement (paper uses pairwise)
                if num_qubits > 1:
                    for q in range(0, num_qubits-1, 2):
                        qml.CNOT(wires=[q, q+1])
                    for q in range(1, num_qubits-1, 2):
                        qml.CNOT(wires=[q, q+1])
            return qml.state()
        
        self.circuit = circuit
    
    def forward(self):
        """Generate a state vector (pure state)"""
        state = self.circuit(self.params)
        return state / torch.norm(state)  # Ensure normalization

## Quantum Discriminator
class QuantumDiscriminator(nn.Module):
    def __init__(self, num_qubits, num_layers):
        super().__init__()
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.params = nn.Parameter(
            torch.rand(num_layers * num_qubits * 3, dtype=torch.float64) * 2 * np.pi
        )
        
        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(params):
            # Apply discriminator circuit to prepare |ψ_d⟩
            for l in range(num_layers):
                for q in range(num_qubits):
                    idx = l*num_qubits*3 + q*3
                    qml.Rot(params[idx], params[idx+1], params[idx+2], wires=q)
                # Entanglement
                if num_qubits > 1:
                    for q in range(0, num_qubits-1, 2):
                        qml.CNOT(wires=[q, q+1])
            return qml.state()
        
        self.circuit = circuit
    
    def forward(self, state_vector, target_state):
        """
        Differentiable implementation of paper's POVM
        Maintains gradient flow throughout
        """
        # Prepare discriminator's state |ψ_d⟩ (with gradient tracking)
        psi_d = self.circuit(self.params)
        psi_d = psi_d / torch.norm(psi_d)
        
        # Normalize states
        state_vector = state_vector / torch.norm(state_vector)
        target_state = target_state / torch.norm(target_state)
        
        # Compute fidelities (keep in computation graph)
        F_target = torch.abs(torch.vdot(target_state, state_vector))**2
        F_psi_d = torch.abs(torch.vdot(psi_d, state_vector))**2
        
        # Compute POVM probability with numerical stability
        numerator = F_psi_d - F_target * F_psi_d
        denominator = 1 - F_target
        
        # Handle cases where denominator is near zero
        safe_denominator = torch.where(denominator < 1e-12, 1e-12, denominator)
        
        # Compute P_fake with gradient tracking
        p_fake = numerator / safe_denominator
        
        # Clamp to valid probability range [0, 1]
        p_fake = torch.clamp(p_fake, 0.0, 1.0)
        
        # For states very close to target, force 0.5 probability
        p_fake = torch.where(F_target > 0.9999, 0.5, p_fake)
        
        return p_fake

## Quantum GAN following paper methodology
class PaperQuantumGAN(nn.Module):
    def __init__(self, num_qubits=2, num_layers=3):
        super().__init__()
        self.generator = QuantumGenerator(num_qubits, num_layers)
        self.discriminator = QuantumDiscriminator(num_qubits, num_layers)
        self.target_state = self.create_target_state()
    
    def create_target_state(self):
        """Create fixed target state |φ⟩ as in paper"""
        # Create a random pure state
        real_part = np.random.randn(2**num_qubits).astype(np.float64)
        imag_part = np.random.randn(2**num_qubits).astype(np.float64)
        state = (real_part + 1j * imag_part).astype(np.complex128)
        state /= np.linalg.norm(state)
        return torch.tensor(state, dtype=torch.complex128)
    
    def train(self, epochs=200, batch_size=1, lr=0.01):
        opt_gen = optim.Adam(self.generator.parameters(), lr=lr)
        opt_disc = optim.Adam(self.discriminator.parameters(), lr=lr)
        
        disc_losses = []
        gen_losses = []
        fidelities = []
        
        for epoch in tqdm(range(epochs)):

            # 1. Train Discriminator
            opt_disc.zero_grad()
            disc_loss = 0
            
            # Process real states (target)
            real_output = self.discriminator(self.target_state, self.target_state)
            # Discriminator should output low for real states
            disc_loss += torch.log(1 - real_output + 1e-12)
            
            # Process fake states (generated)
            fake_state = self.generator().detach()
            fake_output = self.discriminator(fake_state, self.target_state)
            # Discriminator should output high for fake states
            disc_loss += torch.log(fake_output + 1e-12)
            
            disc_loss = -disc_loss / 2  # Negative log loss
            disc_loss.backward()
            opt_disc.step()
            
            # 2. Train Generator
            opt_gen.zero_grad()
            gen_loss = 0
            
            # Generate new fake state
            fake_state = self.generator()
            output = self.discriminator(fake_state, self.target_state)
            # Generator wants discriminator to think fakes are real
            gen_loss = torch.log(1 - output + 1e-12)
            gen_loss = -gen_loss  # Negative log loss
            gen_loss.backward()
            opt_gen.step()
            
            # Store losses
            disc_losses.append(disc_loss.item())
            gen_losses.append(gen_loss.item())
            
            # Calculate fidelity with target state (paper's metric)
            gen_state = self.generator().detach()
            fid = torch.abs(torch.vdot(self.target_state, gen_state))**2
            fidelities.append(fid.item())
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch:3d} | D_loss: {disc_loss.item():.6f} | G_loss: {gen_loss.item():.6f}")
                print(f"  Fidelity: {fid.item():.6f}")

        # Plot results
        plt.figure(figsize=(15, 5))
        
        # Loss plot
        plt.subplot(1, 2, 1)
        plt.plot(disc_losses, label="Discriminator Loss")
        plt.plot(gen_losses, label="Generator Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.legend()
        plt.grid(True)
        
        # Fidelity plot
        plt.subplot(1, 2, 2)
        plt.plot(fidelities)
        plt.xlabel("Epoch")
        plt.ylabel("Fidelity")
        plt.title("State Fidelity with Target")
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig("paper_qgan_results.png")
        plt.show()
        
        # Save final states
        final_state = self.generator().detach().numpy()
        np.save("target_state.npy", self.target_state.numpy())
        np.save("generated_state.npy", final_state)
        
        return disc_losses, gen_losses, fidelities

## Run the Quantum GAN as per paper
if __name__ == "__main__":
    print("Initializing Quantum GAN with Paper's Methodology...")
    qgan = PaperQuantumGAN(num_qubits=num_qubits, num_layers=num_layers)
    
    print("Starting training...")
    losses = qgan.train(epochs=200, batch_size=64, lr=1e-2)  # Batch_size=1 as in paper
    
    print("Training completed!")
    
    # Compare final states
    target_state = np.load("target_state.npy")
    generated_state = np.load("generated_state.npy")
    
    # Convert to density matrices
    rho_target = state_to_density(target_state)
    rho_generated = state_to_density(generated_state)
    
    # Visualize density matrices
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    
    # Target density
    im0 = ax[0].imshow(np.abs(rho_target), cmap='viridis')
    ax[0].set_title("Target Density Matrix")
    plt.colorbar(im0, ax=ax[0])
    
    # Generated density
    im1 = ax[1].imshow(np.abs(rho_generated), cmap='viridis')
    ax[1].set_title("Generated Density Matrix")
    plt.colorbar(im1, ax=ax[1])
    
    plt.tight_layout()
    plt.savefig("density_matrix_comparison.png")
    plt.show()