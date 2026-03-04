import random
from typing import List

import random
from typing import List


# from scene.cameras import Camera

import random
from typing import List


# from scene.cameras import Camera # Assuming your Camera class is available

class HybridBatchSampler:
    """
    A stateful batch sampler with three key features:
    1.  Builds batches from a mix of random and nearest-neighbor views.
    2.  Shuffles neighbors to avoid repetitive patterns.
    3.  Applies camera dropout for regularization.
    Ensures each camera is processed exactly once per epoch.
    """

    def __init__(self, all_cameras: List, batch_size: int, num_random: int, dropout_prob: float = 0.1,
                 world_size: int = 1, global_rank: int = 0):
        """
        Initializes the sampler.

        Args:
            all_cameras (List[Camera]): The full list of camera objects.
            batch_size (int): The total target number of cameras in a batch.
            num_random (int): The number of random cameras to include.
            dropout_prob (float): Probability of dropping a camera from the batch (0.0 to 1.0).
            world_size (int): Total number of parallel processes (for DDP).
            global_rank (int): The rank of the current process.
        """
        if not all_cameras:
            raise ValueError("Camera list cannot be empty.")
        if num_random > batch_size:
            num_random = batch_size
            # raise ValueError("num_random cannot be greater than batch_size.")
        if not (0.0 <= dropout_prob < 1.0):
            raise ValueError("dropout_prob must be between 0.0 and 1.0.")

        self.all_cameras = all_cameras
        self.num_cameras = len(all_cameras)
        self.batch_size = batch_size
        self.num_random = num_random
        self.num_neighbors = batch_size - num_random
        self.dropout_prob = dropout_prob

        self.world_size = world_size
        self.global_rank = global_rank

        self.anchor_queue = []
        self.processed_in_epoch = set()

        # print(
        #     f"[Sampler] Initialized Hybrid Sampler. Batch Size: {self.batch_size}, Random Views: {self.num_random}, Dropout: {self.dropout_prob * 100}%.")
        self._start_new_epoch()

    def _start_new_epoch(self, reduce_batch_size=False):
        """Resets and shuffles the queue of camera indices for a new epoch."""
        # print(f"[Sampler] Rank {self.global_rank}: A new epoch begins. Shuffling camera queue...")
        indices = list(range(self.num_cameras))
        random.shuffle(indices)
        self.anchor_queue = indices[self.global_rank::self.world_size]
        self.processed_in_epoch.clear()

    def __iter__(self):
        self._start_new_epoch()

        return self


    def __next__(self) -> List[int]:
        """Returns the next hybrid batch of cameras with dropout."""
        initial_anchors = []

        # --- Phase 1: Accumulate Random Anchor Views ---
        while self.anchor_queue and len(initial_anchors) < self.num_random:
            candidate_index = self.anchor_queue.pop(0)
            if candidate_index not in self.processed_in_epoch:
                initial_anchors.append(candidate_index)

        if not initial_anchors:
            raise StopIteration

        # --- Phase 2: Accumulate Neighbor Views from ALL Anchors ---
        final_batch_indices = initial_anchors.copy()
        for idx in initial_anchors:
            self.processed_in_epoch.add(idx)

        if self.num_neighbors > 0:
            for anchor_idx in initial_anchors:
                if len(final_batch_indices) >= self.batch_size:
                    break

                anchor_camera = self.all_cameras[anchor_idx]
                potential_neighbors = anchor_camera.nearest_id.copy()
                random.shuffle(potential_neighbors)

                for neighbor_idx in potential_neighbors:
                    if len(final_batch_indices) >= self.batch_size:
                        break
                    if neighbor_idx not in self.processed_in_epoch:
                        final_batch_indices.append(neighbor_idx)
                        self.processed_in_epoch.add(neighbor_idx)

        # --- Phase 3: Apply Camera Dropout (No changes needed here) ---
        batch_after_dropout_indices = []
        if self.dropout_prob > 0 and len(final_batch_indices) > 1:
            # Always keep the first camera to ensure non-empty batches
            batch_after_dropout_indices.append(final_batch_indices[0])
            for idx in final_batch_indices[1:]:
                if random.random() > self.dropout_prob:
                    batch_after_dropout_indices.append(idx)
        else:
            batch_after_dropout_indices = final_batch_indices

        return [self.all_cameras[i] for i in batch_after_dropout_indices]
    # def __next__(self) -> List:
    #     """Returns the next hybrid batch of cameras with dropout."""
    #     final_batch_indices = []
    #     anchor_for_neighbors = None
    #
    #     # --- Phase 1: Accumulate Random Views ---
    #     while self.anchor_queue and len(final_batch_indices) < self.num_random:
    #         candidate_index = self.anchor_queue.pop(0)
    #         if candidate_index not in self.processed_in_epoch:
    #             final_batch_indices.append(candidate_index)
    #             self.processed_in_epoch.add(candidate_index)
    #             # The last valid random camera will be the anchor for neighbors
    #             anchor_for_neighbors = self.all_cameras[candidate_index]
    #
    #     # --- Phase 2: Accumulate Neighbor Views ---
    #     if anchor_for_neighbors and self.num_neighbors > 0:
    #
    #         # --- NEW: Shuffle neighbors to avoid repetitive patterns ---
    #         potential_neighbors = anchor_for_neighbors.nearest_id.copy()
    #         random.shuffle(potential_neighbors)
    #         # --------------------------------------------------------
    #
    #         for neighbor_idx in potential_neighbors:
    #             if len(final_batch_indices) >= self.batch_size:
    #                 break
    #             if neighbor_idx not in self.processed_in_epoch:
    #                 final_batch_indices.append(neighbor_idx)
    #                 self.processed_in_epoch.add(neighbor_idx)
    #
    #     # If after trying to build a batch, it's empty, the epoch is over.
    #     if not final_batch_indices:
    #         raise StopIteration
    #
    #     # --- Phase 3: Apply Camera Dropout ---
    #     batch_after_dropout_indices = []
    #     if self.dropout_prob > 0 and len(final_batch_indices) > 1:
    #         # Always keep the first camera (the primary anchor) to ensure non-empty batches
    #         batch_after_dropout_indices.append(final_batch_indices[0])
    #         # Apply dropout to the rest of the cameras in the batch
    #         for idx in final_batch_indices[1:]:
    #             if random.random() > self.dropout_prob:
    #                 batch_after_dropout_indices.append(idx)
    #     else:
    #         # If no dropout, just use the full batch
    #         batch_after_dropout_indices = final_batch_indices
    #     # ------------------------------------
    #
    #     return [self.all_cameras[i] for i in batch_after_dropout_indices]

