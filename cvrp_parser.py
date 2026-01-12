"""
CVRP Instance Parser for CVRPLib format

This module provides functionality to parse CVRP (Capacitated Vehicle Routing Problem)
instances from the CVRPLib format.
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict
import math


@dataclass
class CVRPInstance:
    """Represents a CVRP instance with all its data."""

    name: str
    comment: str
    type: str
    dimension: int
    edge_weight_type: str
    capacity: int
    node_coords: Dict[int, Tuple[float, float]]  # node_id -> (x, y)
    demands: Dict[int, int]  # node_id -> demand
    depots: List[int]  # List of depot node IDs

    def get_distance(self, node_i: int, node_j: int) -> float:
        """
        Calculate the distance between two nodes based on edge_weight_type.

        Args:
            node_i: First node ID
            node_j: Second node ID

        Returns:
            Distance between the two nodes
        """
        if self.edge_weight_type == "EUC_2D":
            x1, y1 = self.node_coords[node_i]
            x2, y2 = self.node_coords[node_j]
            # Euclidean distance, rounded to nearest integer as per TSPLIB convention
            distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            return round(distance)
        else:
            raise NotImplementedError(f"Edge weight type {self.edge_weight_type} not implemented")

    def get_customers(self) -> List[int]:
        """
        Get list of customer nodes (excluding depots).

        Returns:
            List of customer node IDs
        """
        return [node_id for node_id in self.node_coords.keys() if node_id not in self.depots]

    def __repr__(self) -> str:
        return (f"CVRPInstance(name='{self.name}', dimension={self.dimension}, "
                f"capacity={self.capacity}, depots={self.depots})")


class CVRPParser:
    """Parser for CVRP instances in CVRPLib format."""

    @staticmethod
    def parse(file_path: str) -> CVRPInstance:
        """
        Parse a CVRP instance file.

        Args:
            file_path: Path to the .vrp file

        Returns:
            CVRPInstance object with all the parsed data

        Raises:
            ValueError: If the file format is invalid
        """
        with open(file_path, 'r') as f:
            lines = [line.strip() for line in f.readlines()]

        # Initialize variables
        name = ""
        comment = ""
        type_ = ""
        dimension = 0
        edge_weight_type = ""
        capacity = 0
        node_coords = {}
        demands = {}
        depots = []

        # State machine for parsing
        section = None

        for line in lines:
            if not line or line.startswith("EOF"):
                continue

            # Parse header fields
            if line.startswith("NAME"):
                name = line.split(":", 1)[1].strip()
            elif line.startswith("COMMENT"):
                comment = line.split(":", 1)[1].strip().strip('"')
            elif line.startswith("TYPE"):
                type_ = line.split(":", 1)[1].strip()
            elif line.startswith("DIMENSION"):
                dimension = int(line.split(":", 1)[1].strip())
            elif line.startswith("EDGE_WEIGHT_TYPE"):
                edge_weight_type = line.split(":", 1)[1].strip()
            elif line.startswith("CAPACITY"):
                capacity = int(line.split(":", 1)[1].strip())

            # Section markers
            elif line == "NODE_COORD_SECTION":
                section = "NODE_COORD"
            elif line == "DEMAND_SECTION":
                section = "DEMAND"
            elif line == "DEPOT_SECTION":
                section = "DEPOT"

            # Parse section data
            elif section == "NODE_COORD":
                parts = line.split()
                if len(parts) >= 3:
                    node_id = int(parts[0])
                    x = float(parts[1])
                    y = float(parts[2])
                    node_coords[node_id] = (x, y)

            elif section == "DEMAND":
                parts = line.split()
                if len(parts) >= 2:
                    node_id = int(parts[0])
                    demand = int(parts[1])
                    demands[node_id] = demand

            elif section == "DEPOT":
                depot_id = int(line)
                if depot_id == -1:
                    section = None
                else:
                    depots.append(depot_id)

        # Validate that we have all required data
        if not name:
            raise ValueError("Missing NAME field")
        if dimension == 0:
            raise ValueError("Missing or invalid DIMENSION field")
        if capacity == 0:
            raise ValueError("Missing or invalid CAPACITY field")
        if not node_coords:
            raise ValueError("Missing NODE_COORD_SECTION")
        if not demands:
            raise ValueError("Missing DEMAND_SECTION")
        if not depots:
            raise ValueError("Missing DEPOT_SECTION")

        return CVRPInstance(
            name=name,
            comment=comment,
            type=type_,
            dimension=dimension,
            edge_weight_type=edge_weight_type,
            capacity=capacity,
            node_coords=node_coords,
            demands=demands,
            depots=depots
        )


def load_cvrp_instance(file_path: str) -> CVRPInstance:
    """
    Convenience function to load a CVRP instance from a file.

    Args:
        file_path: Path to the .vrp file

    Returns:
        CVRPInstance object
    """
    return CVRPParser.parse(file_path)


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) > 1:
        instance_path = sys.argv[1]
        instance = load_cvrp_instance(instance_path)

        print(f"Instance: {instance.name}")
        print(f"Comment: {instance.comment}")
        print(f"Type: {instance.type}")
        print(f"Dimension: {instance.dimension}")
        print(f"Edge Weight Type: {instance.edge_weight_type}")
        print(f"Capacity: {instance.capacity}")
        print(f"Depots: {instance.depots}")
        print(f"Number of customers: {len(instance.get_customers())}")

        # Example: Calculate distance between depot and first customer
        depot = instance.depots[0]
        customers = instance.get_customers()
        if customers:
            first_customer = customers[0]
            distance = instance.get_distance(depot, first_customer)
            print(f"\nDistance from depot {depot} to customer {first_customer}: {distance}")
            print(f"Demand at customer {first_customer}: {instance.demands[first_customer]}")
    else:
        print("Usage: python cvrp_parser.py <path_to_vrp_file>")
