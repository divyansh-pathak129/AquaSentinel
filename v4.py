import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage
import random

def load_and_preprocess_maps(flood_map_path, segmentation_map_path):
    """Load and preprocess flood and segmentation maps, ensuring they have the same dimensions."""
    # Load maps
    flood_map = cv2.imread(flood_map_path)
    segmentation_map = cv2.imread(segmentation_map_path)
    
    # Check if maps loaded successfully
    if flood_map is None or segmentation_map is None:
        raise ValueError("Failed to load input maps")
    
    # Get dimensions
    flood_height, flood_width = flood_map.shape[:2]
    seg_height, seg_width = segmentation_map.shape[:2]
    
    print(f"Original dimensions - Flood map: {flood_height}x{flood_width}, Segmentation map: {seg_height}x{seg_width}")
    
    # Resize maps to match if dimensions are different
    if (flood_height != seg_height) or (flood_width != seg_width):
        print("Maps have different dimensions. Resizing...")
        
        # Choose the larger dimensions for better preservation of details
        target_height = max(flood_height, seg_height)
        target_width = max(flood_width, seg_width)
        
        # Resize both maps to the target dimensions
        flood_map = cv2.resize(flood_map, (target_width, target_height), interpolation=cv2.INTER_CUBIC)
        segmentation_map = cv2.resize(segmentation_map, (target_width, target_height), interpolation=cv2.INTER_CUBIC)
        
        print(f"Resized both maps to {target_height}x{target_width}")
    
    # Extract blue channel from flood map (higher values = higher flood risk)
    # The blue channel extraction depends on your flood map's format
    # Option 1: Extract from the blue channel (if flood risk is represented as blue)
    flood_risk = flood_map[:, :, 0].astype(np.float32)
    
    # Option 2: Alternative - create a weighted sum if all channels contain flood information
    # flood_risk = (0.3 * flood_map[:, :, 0] + 0.59 * flood_map[:, :, 1] + 0.11 * flood_map[:, :, 2]).astype(np.float32)
    
    # Normalize flood risk to range 0-1 where 1 is highest risk
    flood_risk = flood_risk / 255.0
    
    # Display the extracted flood risk for verification
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Flood Map")
    plt.imshow(cv2.cvtColor(flood_map, cv2.COLOR_BGR2RGB))
    plt.subplot(1, 2, 2)
    plt.title("Extracted Flood Risk (Darker = Lower Risk)")
    plt.imshow(flood_risk, cmap='Blues_r')
    plt.colorbar(label='Flood Risk')
    plt.tight_layout()
    plt.savefig("flood_risk_extraction.png")
    plt.close()
    
    return flood_map, segmentation_map, flood_risk

def identify_settlement_areas(flood_risk, segmentation_map, risk_threshold=0.3, adaptive=True, max_threshold=0.8):
    """Identify areas suitable for settlement based on flood risk and segmentation."""
    height, width = flood_risk.shape
    
    # If adaptive is True, we'll try increasing thresholds until we find suitable areas
    if adaptive:
        # Start with the provided threshold and gradually increase if needed
        current_threshold = risk_threshold
        step = 0.05
        
        while current_threshold <= max_threshold:
            print(f"Trying risk threshold: {current_threshold:.2f}")
            
            # Create binary mask where 1 = suitable for settlement
            # Areas with flood risk below threshold are suitable
            low_risk_mask = flood_risk < current_threshold
            
            # Generate histogram of flood risk values for analysis
            hist, bins = np.histogram(flood_risk.flatten(), bins=20, range=(0, 1))
            percent_below_threshold = np.sum(flood_risk < current_threshold) / (height * width) * 100
            print(f"Percentage of map below threshold: {percent_below_threshold:.2f}%")
            
            # Try different water body detection approaches
            water_masks = []
            
            # Approach 1: Using blue channel threshold (assuming water is blue)
            water_mask1 = segmentation_map[:, :, 0] > 200
            water_masks.append(("Blue channel > 200", water_mask1))
            
            # Approach 2: Using color ranges (assuming water is blue-ish)
            lower_blue = np.array([100, 0, 0])
            upper_blue = np.array([255, 100, 100])
            water_mask2 = cv2.inRange(segmentation_map, lower_blue, upper_blue) > 0
            water_masks.append(("Blue color range", water_mask2))
            
            # Approach 3: Alternative threshold
            water_mask3 = segmentation_map[:, :, 0] > 150
            water_masks.append(("Blue channel > 150", water_mask3))
            
            # Try each water mask approach
            for mask_name, water_mask in water_masks:
                print(f"Trying water detection method: {mask_name}")
                
                # Areas must be both low flood risk and not water bodies
                suitable_mask = low_risk_mask & ~water_mask
                
                # Apply morphological operations to smooth the mask
                suitable_mask = ndimage.binary_opening(suitable_mask, structure=np.ones((3, 3)))
                suitable_mask = ndimage.binary_closing(suitable_mask, structure=np.ones((3, 3)))
                
                # Calculate percentage of suitable area
                suitable_percent = np.sum(suitable_mask) / (height * width) * 100
                print(f"Suitable area with {mask_name}: {suitable_percent:.2f}%")
                
                # Find connected regions
                labeled_mask, num_regions = ndimage.label(suitable_mask)
                
                if num_regions > 0:
                    # Get region sizes
                    region_sizes = ndimage.sum(suitable_mask, labeled_mask, range(1, num_regions + 1))
                    
                    # Get largest region size as percentage of total area
                    largest_size_percent = np.max(region_sizes) / (height * width) * 100
                    print(f"Largest region: {largest_size_percent:.2f}% of total area")
                    
                    # Check if any region is large enough (at least 1% of total area)
                    if largest_size_percent >= 1.0:
                        print(f"Found suitable region with {mask_name} at threshold {current_threshold:.2f}")
                        largest_region_label = np.argmax(region_sizes) + 1
                        largest_region = (labeled_mask == largest_region_label)
                        
                        # Visualize the suitable areas
                        plt.figure(figsize=(15, 10))
                        plt.subplot(2, 2, 1)
                        plt.title("Flood Risk Map (Darker = Lower Risk)")
                        plt.imshow(flood_risk, cmap='Blues_r')
                        plt.colorbar()
                        
                        plt.subplot(2, 2, 2)
                        plt.title("Segmentation Map")
                        plt.imshow(cv2.cvtColor(segmentation_map, cv2.COLOR_BGR2RGB))
                        
                        plt.subplot(2, 2, 3)
                        plt.title(f"Water Bodies ({mask_name})")
                        plt.imshow(water_mask, cmap='gray')
                        
                        plt.subplot(2, 2, 4)
                        plt.title(f"Selected Settlement Area (Threshold: {current_threshold:.2f})")
                        plt.imshow(largest_region, cmap='viridis')
                        
                        plt.tight_layout()
                        plt.savefig("settlement_area_selection.png")
                        plt.close()
                        
                        return largest_region
            
            # If we get here, none of the water detection methods worked with the current threshold
            current_threshold += step
    
    # Non-adaptive approach or if adaptive approach failed
    # Create binary mask where 1 = suitable for settlement
    low_risk_mask = flood_risk < risk_threshold
    
    # Identify water bodies from segmentation map (simplistic approach)
    # You may need to adjust this based on your segmentation map representation
    water_mask = segmentation_map[:, :, 0] > 180
    
    # Areas must be both low flood risk and not water bodies
    suitable_mask = low_risk_mask & ~water_mask
    
    # Apply morphological operations to smooth the mask
    suitable_mask = ndimage.binary_opening(suitable_mask, structure=np.ones((3, 3)))
    
    # Find connected regions
    labeled_mask, num_regions = ndimage.label(suitable_mask)
    
    # Select the largest connected region
    if num_regions == 0:
        # If no suitable regions found, create a fallback region
        print("WARNING: No suitable regions found. Creating fallback area in lowest risk region.")
        
        # Find the area with the lowest average flood risk
        kernel_size = 50  # Size of area to consider
        lowest_risk = float('inf')
        best_y, best_x = 0, 0
        
        # Sample the image at regular intervals to speed up processing
        step = max(1, kernel_size // 4)
        
        for y in range(0, height - kernel_size, step):
            for x in range(0, width - kernel_size, step):
                region = flood_risk[y:y+kernel_size, x:x+kernel_size]
                region_risk = np.mean(region)
                
                if region_risk < lowest_risk:
                    lowest_risk = region_risk
                    best_y, best_x = y, x
        
        # Create a circular mask at the best location
        y_indices, x_indices = np.ogrid[:height, :width]
        mask = (x_indices - best_x) ** 2 + (y_indices - best_y) ** 2 <= (kernel_size // 2) ** 2
        
        print(f"Created fallback area at ({best_x}, {best_y}) with radius {kernel_size//2}")
        
        # Visualize the fallback area
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title("Flood Risk Map")
        plt.imshow(flood_risk, cmap='Blues_r')
        plt.colorbar()
        plt.subplot(1, 2, 2)
        plt.title("Fallback Settlement Area")
        plt.imshow(mask, cmap='viridis')
        plt.tight_layout()
        plt.savefig("fallback_settlement_area.png")
        plt.close()
        
        return mask
    
    region_sizes = ndimage.sum(suitable_mask, labeled_mask, range(1, num_regions + 1))
    largest_region_label = np.argmax(region_sizes) + 1
    largest_region = (labeled_mask == largest_region_label)
    
    return largest_region

def generate_settlement_pattern(suitable_area, num_houses=60, min_distance=10):
    """Generate settlement pattern with houses respecting minimum distance."""
    height, width = suitable_area.shape
    houses = []
    
    # Find potential house locations (all suitable pixels)
    potential_locations = np.column_stack(np.where(suitable_area))
    
    if len(potential_locations) == 0:
        raise ValueError("No suitable locations found for houses")
        
    # If we don't have enough suitable locations, adjust the minimum distance
    if len(potential_locations) < num_houses:
        print(f"Warning: Only {len(potential_locations)} suitable locations found. Reducing number of houses.")
        num_houses = min(len(potential_locations), num_houses)
        min_distance = max(1, min_distance // 2)
        print(f"Adjusted to {num_houses} houses with minimum distance {min_distance}")
    
    # Randomly select initial house
    if len(potential_locations) > 0:
        idx = random.randint(0, len(potential_locations) - 1)
        houses.append(potential_locations[idx])
    
    # Simple algorithm to place houses with minimum distance
    attempts = 0
    max_attempts = 2000
    
    while len(houses) < num_houses and attempts < max_attempts:
        attempts += 1
        
        # Randomly select a location
        if len(potential_locations) > 0:
            idx = random.randint(0, len(potential_locations) - 1)
            candidate = potential_locations[idx]
            
            # Check if it's far enough from existing houses
            valid = True
            for house in houses:
                distance = np.sqrt(np.sum((candidate - house) ** 2))
                if distance < min_distance:
                    valid = False
                    break
            
            if valid:
                houses.append(candidate)
                attempts = 0  # Reset attempts counter on success
    
    # If we couldn't place enough houses with the given constraints, relax them
    if len(houses) < num_houses * 0.75:
        print(f"Warning: Could only place {len(houses)} houses. Trying with reduced minimum distance.")
        return generate_settlement_pattern(suitable_area, num_houses, min_distance // 2)
    
    print(f"Generated {len(houses)} houses")
    return np.array(houses)

def design_drainage_system(houses, segmentation_map, flood_risk):
    """Design drainage system around the settlement."""
    height, width = flood_risk.shape
    drainage_map = np.zeros((height, width), dtype=np.uint8)
    
    if len(houses) == 0:
        print("Warning: No houses to create drainage for")
        return drainage_map
    
    # Create a heat map of houses to identify dense areas
    house_density = np.zeros((height, width), dtype=np.float32)
    for house in houses:
        y, x = house
        y_min, y_max = max(0, y-20), min(height, y+20)
        x_min, x_max = max(0, x-20), min(width, x+20)
        house_density[y_min:y_max, x_min:x_max] += 1
    
    # Smooth the density map
    house_density = ndimage.gaussian_filter(house_density, sigma=10)
    
    # Find water bodies from segmentation map (try multiple approaches)
    # Approach 1: Basic blue threshold
    water_bodies1 = segmentation_map[:, :, 0] > 180
    
    # Approach 2: Color range for blue-ish pixels
    lower_blue = np.array([100, 0, 0])
    upper_blue = np.array([255, 100, 100])
    water_bodies2 = cv2.inRange(segmentation_map, lower_blue, upper_blue) > 0
    
    # Combine approaches
    water_bodies = water_bodies1 | water_bodies2
    
    # If no water bodies detected, use the highest flood risk areas as proxy
    if np.sum(water_bodies) < (height * width * 0.01):  # Less than 1% of the map
        print("Warning: No significant water bodies detected. Using high flood risk areas as proxy.")
        water_bodies = flood_risk > 0.7  # Assuming higher values = higher risk
    
    # Create gradient map to guide drainage toward lower elevation (assumes flood risk correlates with elevation)
    gradient_y, gradient_x = np.gradient(flood_risk)
    
    # Generate primary drainage channels
    # 1. Around settlement perimeter
    settlement_mask = np.zeros((height, width), dtype=np.uint8)
    for house in houses:
        y, x = house
        if 0 <= y < height and 0 <= x < width:
            settlement_mask[y, x] = 1
    
    # Dilate to get settlement area
    settlement_area = ndimage.binary_dilation(settlement_mask, structure=np.ones((15, 15)))
    
    # Find perimeter
    perimeter = ndimage.binary_dilation(settlement_area) & ~settlement_area
    
    # Add primary drainage channels around settlement
    drainage_map[perimeter] = 1
    
    # 2. Connect to nearby water bodies
    dilated_water = ndimage.binary_dilation(water_bodies, structure=np.ones((5, 5)))
    
    # If there are water bodies, connect drainage to them
    if np.sum(dilated_water) > 0:
        # Find shortest path to water bodies
        distance_to_water = ndimage.distance_transform_edt(~dilated_water)
        
        # Add channels connecting to water (up to 4 channels)
        num_channels = min(4, np.sum(perimeter) // 100 + 1)  # Scale with perimeter size
        
        for i in range(num_channels):
            # Find points in perimeter closest to water
            perimeter_points = np.column_stack(np.where(perimeter))
            if len(perimeter_points) == 0:
                continue
                
            # Sort by distance to water
            distances = [distance_to_water[y, x] for y, x in perimeter_points]
            sorted_indices = np.argsort(distances)
            
            if len(sorted_indices) > 0:
                closest_point = perimeter_points[sorted_indices[0]]
                
                # Find nearest water point
                water_points = np.column_stack(np.where(dilated_water))
                if len(water_points) > 0:
                    distances = [np.sqrt(np.sum((closest_point - wp) ** 2)) for wp in water_points]
                    nearest_water = water_points[np.argmin(distances)]
                    
                    # Create drainage channel between points
                    y1, x1 = closest_point
                    y2, x2 = nearest_water
                    
                    # Simple line drawing
                    num_points = int(max(abs(y2 - y1), abs(x2 - x1)) * 1.5)
                    for t in range(num_points + 1):
                        alpha = t / num_points
                        y = int((1 - alpha) * y1 + alpha * y2)
                        x = int((1 - alpha) * x1 + alpha * x2)
                        if 0 <= y < height and 0 <= x < width:
                            drainage_map[y, x] = 1
                            # Make drainage wider
                            for dy in range(-2, 3):
                                for dx in range(-2, 3):
                                    ny, nx = y + dy, x + dx
                                    if 0 <= ny < height and 0 <= nx < width:
                                        drainage_map[ny, nx] = 1
                    
                    # Remove this part of perimeter to avoid selecting it again
                    for y in range(max(0, y1-10), min(height, y1+10)):
                        for x in range(max(0, x1-10), min(width, x1+10)):
                            if 0 <= y < height and 0 <= x < width:
                                perimeter[y, x] = 0
    
    # 3. Add secondary drainage within settlement
    for house in houses:
        y, x = house
        if 0 <= y < height and 0 <= x < width:
            # Find direction of steepest gradient (water flow)
            gy, gx = gradient_y[y, x], gradient_x[y, x]
            magnitude = np.sqrt(gy**2 + gx**2)
            if magnitude > 0:
                gy, gx = gy / magnitude, gx / magnitude
            else:
                gy, gx = 0, 0
            
            # Create small drainage channel in gradient direction
            length = random.randint(5, 15)
            for i in range(1, length):
                ny, nx = int(y + i * gy), int(x + i * gx)
                if 0 <= ny < height and 0 <= nx < width:
                    drainage_map[ny, nx] = 1
    
    # Dilate drainage system for visibility
    drainage_map = ndimage.binary_dilation(drainage_map, structure=np.ones((2, 2)))
    
    return drainage_map

def generate_settlement_visualization(flood_map, segmentation_map, houses, drainage_system):
    """Generate final visualization with settlements and drainage."""
    # Create a copy of segmentation map for visualization
    result_map = segmentation_map.copy()
    
    # Add drainage system (red)
    drainage_indices = np.where(drainage_system == 1)
    if len(drainage_indices[0]) > 0:  # Check if drainage exists
        result_map[drainage_indices[0], drainage_indices[1]] = [0, 0, 255]  # Red color (BGR)
    
    # Add houses (brown)
    if len(houses) > 0:  # Check if houses exist
        for house in houses:
            y, x = house
            if 0 <= y < result_map.shape[0] and 0 <= x < result_map.shape[1]:
                # Draw a small square for each house
                house_size = 2
                y_min, y_max = max(0, y-house_size), min(result_map.shape[0], y+house_size+1)
                x_min, x_max = max(0, x-house_size), min(result_map.shape[1], x+house_size+1)
                result_map[y_min:y_max, x_min:x_max] = [30, 65, 155]  # Brown color (BGR)
    
    return result_map

def generate_settlement(flood_map_path, segmentation_map_path, output_path="settlement_plan.png", 
                        risk_threshold=0.3, num_houses_min=50, num_houses_max=70, adaptive=True):
    """Generate settlement with houses and drainage based on flood and segmentation maps."""
    try:
        # Load and preprocess maps
        flood_map, segmentation_map, flood_risk = load_and_preprocess_maps(flood_map_path, segmentation_map_path)
        
        # Identify suitable areas for settlement (with adaptive thresholding)
        suitable_areas = identify_settlement_areas(flood_risk, segmentation_map, risk_threshold, adaptive=adaptive)
        
        # Generate settlement pattern
        houses = generate_settlement_pattern(suitable_areas, num_houses=random.randint(num_houses_min, num_houses_max), min_distance=12)
        
        # Design drainage system
        drainage_system = design_drainage_system(houses, segmentation_map, flood_risk)
        
        # Generate final visualization
        result_map = generate_settlement_visualization(flood_map, segmentation_map, houses, drainage_system)
        
        # Save result
        cv2.imwrite(output_path, result_map)
        
        # Create visualization for display
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 2, 1)
        plt.title("Flood Map")
        plt.imshow(cv2.cvtColor(flood_map, cv2.COLOR_BGR2RGB))
        
        plt.subplot(2, 2, 2)
        plt.title("Segmentation Map")
        plt.imshow(cv2.cvtColor(segmentation_map, cv2.COLOR_BGR2RGB))
        
        plt.subplot(2, 2, 3)
        plt.title("Suitable Areas")
        plt.imshow(suitable_areas, cmap='gray')
        
        plt.subplot(2, 2, 4)
        plt.title("Settlement Plan")
        plt.imshow(cv2.cvtColor(result_map, cv2.COLOR_BGR2RGB))
        
        plt.tight_layout()
        plt.savefig("settlement_visualization.png")
        plt.show()
        
        print(f"Settlement plan generated with {len(houses)} houses and saved to {output_path}")
        return result_map
    
    except Exception as e:
        print(f"Error generating settlement: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# Example usage
if __name__ == "__main__":
    # Replace these paths with your actual file paths
    flood_map_path = "0006_f.png"
    segmentation_map_path = "0006_i2.png"
    
    # Try with adaptive thresholding
    result = generate_settlement(
        flood_map_path, 
        segmentation_map_path, 
        risk_threshold=0.3,
        adaptive=True  # This will automatically try different thresholds
    )
    
    if result is not None:
        print("Settlement generation completed successfully!")
    else:
        print("Settlement generation failed.")