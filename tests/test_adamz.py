
import unittest
import torch
from torch.nn import Linear
from torch.nn.functional import mse_loss
from adamz.adamz import AdamZ

class TestAdamZOptimizer(unittest.TestCase):
    def setUp(self):
        # Set up a simple model and optimizer for testing
        self.model = Linear(10, 1)
        self.optimizer = AdamZ(
            self.model.parameters(),
            lr=0.01,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01,
            overshoot_factor=0.5,
            stagnation_factor=1.2,
            stagnation_threshold=0.1,
            patience=10,
            stagnation_period=5,
            min_lr=1e-5,
            max_lr=0.1,
        )

    def test_initialization(self):
        # Test that the optimizer initializes correctly
        self.assertEqual(self.optimizer.defaults["lr"], 0.01)
        self.assertEqual(self.optimizer.defaults["betas"], (0.9, 0.999))
        self.assertEqual(self.optimizer.defaults["eps"], 1e-8)

    def test_invalid_parameters(self):
        # Test invalid parameter values
        with self.assertRaises(ValueError):
            AdamZ(self.model.parameters(), lr=-0.01)  # Negative learning rate
        with self.assertRaises(ValueError):
            AdamZ(self.model.parameters(), betas=(1.1, 0.999))  # Invalid beta1
        with self.assertRaises(ValueError):
            AdamZ(self.model.parameters(), betas=(0.9, 1.1))  # Invalid beta2
        with self.assertRaises(ValueError):
            AdamZ(self.model.parameters(), overshoot_factor=1.5)  # Invalid overshoot
        with self.assertRaises(ValueError):
            AdamZ(self.model.parameters(), stagnation_factor=0.5)  # Invalid stagnation

    def test_step(self):
        # Test a single optimization step
        inputs = torch.randn(32, 10)
        targets = torch.randn(32, 1)

        # Forward pass
        outputs = self.model(inputs)
        loss = mse_loss(outputs, targets)

        # Backward pass
        loss.backward()
        self.optimizer.step()

        # Check that parameters have been updated
        for param_group in self.optimizer.param_groups:
            self.assertGreater(param_group["lr"], 0)

    def test_adjust_learning_rate(self):
        # Test learning rate adjustment for overshooting and stagnation
        losses = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.6, 0.7, 0.8, 0.9]
        for loss in losses:
            self.optimizer.adjust_learning_rate(loss)

        # Check that learning rates are adjusted within bounds
        for param_group in self.optimizer.param_groups:
            lr = param_group["lr"]
            self.assertGreaterEqual(lr, self.optimizer.min_lr)
            self.assertLessEqual(lr, self.optimizer.max_lr)

    def test_patience_and_stagnation(self):
        # Test that patience and stagnation thresholds work correctly
        losses = [0.5] * 15  # Simulate stagnation
        for loss in losses:
            self.optimizer.adjust_learning_rate(loss)

        for param_group in self.optimizer.param_groups:
            self.assertGreaterEqual(param_group["lr"], self.optimizer.min_lr)
            self.assertLessEqual(param_group["lr"], self.optimizer.max_lr)

    def test_state_dict(self):
        # Test saving and loading state dict
        state_dict = self.optimizer.state_dict()
        new_optimizer = AdamZ(self.model.parameters())
        new_optimizer.load_state_dict(state_dict)

        self.assertEqual(state_dict, new_optimizer.state_dict())

if __name__ == "__main__":
    unittest.main()
