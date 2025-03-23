import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import os

def segmentationmap(segmentation):
    s_map = cv2.imread(segmentation, cv2.IMREAD_COLOR)

    # Handle transparency (remove alpha channel if exists)
    if s_map is None:
        raise ValueError(f"Could not read image at path: {segmentation}")
        
    if len(s_map.shape) > 2 and s_map.shape[-1] == 4:
        s_map = s_map[:, :, :3]  # Convert from BGRA to BGR

    # Extracting height and width to resize the image and maintain aspect ratio
    h, w = s_map.shape[:2]
    ratio = 256/w
    resized_s = cv2.resize(s_map, (256, int(h*ratio)))

    # Creating an array for the segmentation map (shape = 256x256x3)
    seg_map = cv2.cvtColor(resized_s, cv2.COLOR_BGR2RGB)

    return seg_map

def detect_water(seg_map):
    """
    Detect water regions in the segmentation map
    
    Parameters:
    - seg_map: The segmentation map
    
    Returns:
    - water_mask: Boolean mask where True values represent water
    """
    # Water is typically represented by blue shades
    blue_threshold = 1.5  # Blue channel should be this many times higher than red/green
    
    # Calculate blue dominance
    blue_dominant = (seg_map[:,:,2] > seg_map[:,:,0] * blue_threshold) & (seg_map[:,:,2] > seg_map[:,:,1] * blue_threshold)
    
    # Water typically has lower overall brightness than sky
    brightness = np.mean(seg_map, axis=-1)
    low_brightness = brightness < 180
    
    # Combine conditions
    water_mask = blue_dominant & low_brightness
    
    # Clean up with morphological operations
    kernel = np.ones((3, 3), np.uint8)
    water_mask = cv2.morphologyEx(water_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_OPEN, kernel)
    
    return water_mask.astype(bool)

def find_multiple_regions(seg_map, num_regions=3):
    """
    Find multiple suitable land regions for village placement
    
    Parameters:
    - seg_map: The segmentation map
    - num_regions: Number of different regions to find
    
    Returns:
    - regions: List of (region_color, region_mask) tuples
    - land_mask: Boolean mask of all land areas
    """
    # Get image dimensions
    h, w = seg_map.shape[:2]
    
    # Detect water regions
    water_mask = detect_water(seg_map)
    
    # Create land mask (non-water)
    land_mask = ~water_mask
    
    # Get all land pixels
    land_pixels = np.column_stack(np.where(land_mask))
    
    if len(land_pixels) == 0:
        print("No land detected in the image. Using full image.")
        land_mask = np.ones((h, w), dtype=bool)
        land_pixels = np.column_stack(np.where(land_mask))
    
    # Divide the image into grid sections to find diverse regions
    grid_size = 3  # 3x3 grid
    cell_h = h // grid_size
    cell_w = w // grid_size
    
    regions = []
    visited_cells = set()
    
    # First try to find regions in different grid cells
    attempts = 0
    while len(regions) < num_regions and attempts < 20:
        # Select a grid cell that hasn't been visited yet
        if len(visited_cells) < grid_size * grid_size:
            while True:
                grid_y = random.randint(0, grid_size - 1)
                grid_x = random.randint(0, grid_size - 1)
                cell_id = (grid_y, grid_x)
                if cell_id not in visited_cells:
                    visited_cells.add(cell_id)
                    break
        else:
            # All cells visited, pick a random one
            grid_y = random.randint(0, grid_size - 1)
            grid_x = random.randint(0, grid_size - 1)
        
        # Define the cell boundaries
        cell_top = grid_y * cell_h
        cell_left = grid_x * cell_w
        cell_bottom = min((grid_y + 1) * cell_h, h)
        cell_right = min((grid_x + 1) * cell_w, w)
        
        # Get land pixels in this cell
        cell_land_mask = land_mask[cell_top:cell_bottom, cell_left:cell_right]
        if np.sum(cell_land_mask) < 100:  # Not enough land in this cell
            attempts += 1
            continue
            
        cell_land_pixels = np.column_stack(np.where(cell_land_mask))
        if len(cell_land_pixels) == 0:
            attempts += 1
            continue
            
        # Adjust coordinates to full image space
        cell_land_pixels[:, 0] += cell_top
        cell_land_pixels[:, 1] += cell_left
        
        # Sample several points in this cell
        sample_points = []
        for _ in range(10):
            idx = random.randint(0, len(cell_land_pixels) - 1)
            y, x = cell_land_pixels[idx]
            sample_points.append((y, x))
        
        if not sample_points:
            attempts += 1
            continue
            
        # Get colors at sample points
        region_colors = [seg_map[y, x] for y, x in sample_points]
        
        # Count occurrences of each color
        color_counts = {}
        for color in region_colors:
            color_tuple = tuple(color)
            if color_tuple in color_counts:
                color_counts[color_tuple] += 1
            else:
                color_counts[color_tuple] = 1
        
        if not color_counts:
            attempts += 1
            continue
            
        # Select the most common color in this cell
        most_common = max(color_counts.items(), key=lambda x: x[1])
        region_color = np.array(most_common[0])
        
        # Create mask for the selected region
        region_mask = np.all(seg_map == region_color, axis=-1)
        
        # Limit the region to this cell to avoid overlapping regions
        cell_mask = np.zeros((h, w), dtype=bool)
        cell_mask[cell_top:cell_bottom, cell_left:cell_right] = True
        region_mask = region_mask & cell_mask
        
        # Ensure region is on land
        region_mask = region_mask & land_mask
        
        # Check if the region is large enough
        if np.sum(region_mask) < 100:
            # Try colors that are within a certain range, but still on land
            expanded_mask = np.zeros((h, w), dtype=bool)
            for dy in range(-20, 21, 5):
                for dx in range(-20, 21, 5):
                    similar_color = np.clip(region_color + np.array([dy, dy, dx]), 0, 255)
                    similar_mask = np.all(np.abs(seg_map.astype(int) - similar_color.astype(int)) < 30, axis=-1)
                    expanded_mask = expanded_mask | similar_mask
            
            # Only use expanded mask if it's substantially larger
            expanded_mask = expanded_mask & land_mask & cell_mask
            if np.sum(expanded_mask) > 100:
                region_mask = expanded_mask
            else:
                attempts += 1
                continue
        
        # Make sure this region doesn't overlap too much with existing regions
        overlap = False
        for _, existing_mask in regions:
            if np.sum(region_mask & existing_mask) > 0.5 * np.sum(region_mask):
                overlap = True
                break
        
        if not overlap:
            regions.append((region_color, region_mask))
        
        attempts += 1
    
    # If we still don't have enough regions, try to find more anywhere on land
    if len(regions) < num_regions:
        existing_masks = np.zeros((h, w), dtype=bool)
        for _, mask in regions:
            existing_masks = existing_masks | mask
        
        # Find more regions in remaining land areas
        remaining_attempts = 0
        while len(regions) < num_regions and remaining_attempts < 20:
            # Sample random land points not in existing regions
            available_land = land_mask & ~existing_masks
            if np.sum(available_land) < 100:
                break  # Not enough available land
                
            available_pixels = np.column_stack(np.where(available_land))
            if len(available_pixels) == 0:
                break
                
            sample_points = []
            for _ in range(10):
                idx = random.randint(0, len(available_pixels) - 1)
                y, x = available_pixels[idx]
                sample_points.append((y, x))
            
            # Get colors at sample points
            region_colors = [seg_map[y, x] for y, x in sample_points]
            
            # Count occurrences of each color
            color_counts = {}
            for color in region_colors:
                color_tuple = tuple(color)
                if color_tuple in color_counts:
                    color_counts[color_tuple] += 1
                else:
                    color_counts[color_tuple] = 1
            
            if not color_counts:
                remaining_attempts += 1
                continue
                
            # Select the most common color
            most_common = max(color_counts.items(), key=lambda x: x[1])
            region_color = np.array(most_common[0])
            
            # Create mask for the selected region
            region_mask = np.all(seg_map == region_color, axis=-1)
            
            # Ensure region is on land and not in existing regions
            region_mask = region_mask & land_mask & ~existing_masks
            
            # Check if the region is large enough
            if np.sum(region_mask) < 100:
                # Try slightly expanding the region
                expanded_mask = np.zeros((h, w), dtype=bool)
                for dy in range(-20, 21, 5):
                    for dx in range(-20, 21, 5):
                        similar_color = np.clip(region_color + np.array([dy, dy, dx]), 0, 255)
                        similar_mask = np.all(np.abs(seg_map.astype(int) - similar_color.astype(int)) < 30, axis=-1)
                        expanded_mask = expanded_mask | similar_mask
                
                # Only use expanded mask if it's substantially larger
                expanded_mask = expanded_mask & land_mask & ~existing_masks
                if np.sum(expanded_mask) > 100:
                    region_mask = expanded_mask
                else:
                    remaining_attempts += 1
                    continue
            
            regions.append((region_color, region_mask))
            existing_masks = existing_masks | region_mask
            remaining_attempts += 1
    
    print(f"Found {len(regions)} suitable regions for villages")
    return regions, land_mask

def create_dense_cluster(center_y, center_x, region_mask, land_mask, modified_map, num_houses, spacing_range=(1, 3), house_color=np.array([128, 0, 128])):
    """
    Create a dense cluster of houses around a center point
    
    Parameters:
    - center_y, center_x: Center coordinates of the cluster
    - region_mask: Boolean mask of the region
    - land_mask: Boolean mask of all land areas
    - modified_map: Map to modify
    - num_houses: Number of houses to place
    - spacing_range: Min/max spacing between houses
    - house_color: Color for houses
    
    Returns:
    - placed_houses: List of (y, x) coordinates where houses were placed
    """
    h, w = modified_map.shape[:2]
    placed_houses = []
    attempts = 0
    max_attempts = 2000
    
    while len(placed_houses) < num_houses and attempts < max_attempts:
        # Determine how to select the next candidate
        if random.random() < 0.85 or not placed_houses:  # 85% chance to place near existing houses
            if placed_houses:
                # Select a reference house
                ref_idx = random.randint(0, len(placed_houses) - 1)
                ref_y, ref_x = placed_houses[ref_idx]
                
                # Create a new candidate near this house
                angle = random.uniform(0, 2 * np.pi)
                if random.random() < 0.7:
                    distance = random.randint(2, 5)  # Shorter distances
                else:
                    distance = random.randint(1, 3)  # Very short for tight clusters
                
                # Calculate position
                new_x = int(ref_x + distance * np.cos(angle))
                new_y = int(ref_y + distance * np.sin(angle))
            else:
                # No houses yet, place near center
                angle = random.uniform(0, 2 * np.pi)
                distance = random.randint(1, 3)  # First houses close to center
                
                new_x = int(center_x + distance * np.cos(angle))
                new_y = int(center_y + distance * np.sin(angle))
        else:
            # Place a house somewhere else in the region
            angle = random.uniform(0, 2 * np.pi)
            distance = random.randint(5, 15)  # Further but still in region
            
            new_x = int(center_x + distance * np.cos(angle))
            new_y = int(center_y + distance * np.sin(angle))
        
        # Check bounds
        if not (0 <= new_y < h and 0 <= new_x < w):
            attempts += 1
            continue
            
        # Check if in region and on land
        if not (region_mask[new_y, new_x] and land_mask[new_y, new_x]):
            # Try to find a nearby valid point
            valid = False
            search_radius = 2
            while not valid and search_radius < 8:
                for dy in range(-search_radius, search_radius+1):
                    for dx in range(-search_radius, search_radius+1):
                        search_y, search_x = new_y + dy, new_x + dx
                        if (0 <= search_y < h and 0 <= search_x < w and
                            region_mask[search_y, search_x] and land_mask[search_y, search_x]):
                            new_y, new_x = search_y, search_x
                            valid = True
                            break
                    if valid:
                        break
                search_radius += 1
            
            if not valid:
                attempts += 1
                continue
        
        # Check spacing constraint
        too_close = False
        for house_y, house_x in placed_houses:
            distance = np.sqrt((house_x - new_x)**2 + (house_y - new_y)**2)
            min_spacing = random.randint(spacing_range[0], spacing_range[1])
            
            if distance < min_spacing:
                too_close = True
                break
        
        if not too_close:
            placed_houses.append((new_y, new_x))
            modified_map[new_y, new_x] = house_color
        
        attempts += 1
    
    return placed_houses

def add_multiple_villages(seg_map, total_houses=None, villages=3, spacing_range=(1, 3)):
    """
    Add multiple villages across the map, each with a dense cluster of houses
    
    Parameters:
    - seg_map: The original segmentation map
    - total_houses: Total number of houses to add (between 80-150 if None)
    - villages: Number of village clusters to create
    - spacing_range: Range of pixel spacing between houses
    
    Returns:
    - Modified segmentation map with houses
    """
    house_spacing = (2, 5)
    # Create a copy to avoid modifying the original
    modified_map = seg_map.copy()
    
    # Find multiple suitable regions for villages
    regions, land_mask = find_multiple_regions(seg_map, num_regions=villages)
    
    # If we couldn't find enough regions, adjust the number of villages
    num_villages = len(regions)
    if num_villages == 0:
        print("No suitable regions found. Cannot add villages.")
        return modified_map
    
    # Determine total number of houses if not specified
    if total_houses is None:
        total_houses = random.randint(80, 150)
    
    # Purple color for houses
    house_color = np.array([128, 0, 128])  # Purple in RGB
    
    # Distribute houses across villages, with varying sizes
    houses_per_village = []
    remaining_houses = total_houses
    
    # Assign houses to each village with some variation in size
    for i in range(num_villages):
        if i == num_villages - 1:
            # Last village gets all remaining houses
            houses_per_village.append(remaining_houses)
        else:
            # Varies between 30% and 50% of remaining houses
            proportion = random.uniform(0.3, 0.5)
            houses = max(10, int(remaining_houses * proportion))
            houses_per_village.append(houses)
            remaining_houses -= houses
    
    # Shuffle to randomize the order (so the larger villages aren't always in the same places)
    combined = list(zip(regions, houses_per_village))
    random.shuffle(combined)
    regions, houses_per_village = zip(*combined)
    
    # Create debug map for visualizing regions
    debug_map = modified_map.copy()
    colors = [
        np.array([255, 0, 0]),   # Red
        np.array([0, 255, 0]),   # Green
        np.array([0, 0, 255]),   # Blue
        np.array([255, 255, 0]), # Yellow
        np.array([255, 0, 255]), # Magenta
        np.array([0, 255, 255])  # Cyan
    ]
    
    # Visualize each region with a different color
    for i, (_, region_mask) in enumerate(regions):
        color_idx = i % len(colors)
        region_debug = debug_map.copy()
        region_debug[region_mask] = region_debug[region_mask] * 0.7 + colors[color_idx] * 0.3
        
        # Save the debug image
        # save_image(region_debug, f"debug_region_{i+1}.png")
    
    # Place houses in each village
    all_placed_houses = []
    
    for i, ((region_color, region_mask), num_houses) in enumerate(zip(regions, houses_per_village)):
        print(f"Creating village {i+1} with {num_houses} houses")
        
        # Calculate centroid of the region as village center
        region_y_coords, region_x_coords = np.where(region_mask)
        if len(region_y_coords) > 0:
            # Calculate centroid of the region
            center_y = int(np.mean(region_y_coords))
            center_x = int(np.mean(region_x_coords))
            
            # Make sure the center point is actually in the region
            if not region_mask[center_y, center_x]:
                # Find closest point in region
                distances = np.sqrt((region_y_coords - center_y)**2 + (region_x_coords - center_x)**2)
                closest_idx = np.argmin(distances)
                center_y, center_x = region_y_coords[closest_idx], region_x_coords[closest_idx]
        else:
            # Fallback to random point in region
            region_pixels = np.column_stack(np.where(region_mask))
            if len(region_pixels) > 0:
                idx = random.randint(0, len(region_pixels) - 1)
                center_y, center_x = region_pixels[idx]
            else:
                print(f"Village {i+1}: Empty region, skipping")
                continue
        
        # Create a dense cluster of houses in this village
        placed_houses = create_dense_cluster(
            center_y, center_x, region_mask, land_mask, 
            modified_map, num_houses, house_spacing, house_color
        )
        
        all_placed_houses.extend(placed_houses)
        print(f"Village {i+1}: Placed {len(placed_houses)} out of {num_houses} houses")
    
    print(f"Added {len(all_placed_houses)} houses out of {total_houses} requested across {num_villages} villages")
    return modified_map

def save_image(image, filename):
    """Save an image to the current directory"""
    # Convert from RGB to BGR for cv2.imwrite
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, image_bgr)
    print(f"Saved image to {os.path.abspath(filename)}")

# Example usage
image_path = "E:\\flood_detect\\final_maps\data_set\\0050_i2.png"  # Provide the correct path

try:
    # Get the original segmentation map
    original_seg_map = segmentationmap(image_path)
    
    # Add multiple villages with houses distributed across them
    total_houses = random.randint(75, 100)  # Increased total houses
    num_villages = random.randint(1, 1)      # Create 3-5 village clusters
    
    modified_seg_map = add_multiple_villages(
        original_seg_map, 
        total_houses=total_houses, 
        villages=num_villages, 
        spacing_range=(1, 3)
    )
    
    # Save both images to the current directory
    # save_image(original_seg_map, "original_segmentation.png")
    # save_image(modified_seg_map, f"segmentation_with_multiple_villages.png")
    save_image(modified_seg_map, f"0050_v.png")
    
    # Count actual number of houses placed
    house_pixels = np.where(np.all(modified_seg_map == [128, 0, 128], axis=-1))
    house_count = len(house_pixels[0])
    
    # Display the original and modified maps side by side
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(original_seg_map)
    plt.title("Original Segmentation Map")
    plt.axis("off")
    
    plt.subplot(1, 2, 2)
    plt.imshow(modified_seg_map)
    plt.title(f"Map with {house_count} Houses in {num_villages} Villages")
    plt.axis("off")
    
    plt.tight_layout()
    plt.show()
    
except Exception as e:
    print(f"Error: {e}")