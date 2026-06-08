import pygame
import math
import sys
import numpy as np

def animate_pendulum(theta, L, T):
    """
    Animates the pendulum motion based on the numerical solution.
    
    Args:
        theta (np.array): The full array of angles (should include alpha at index 0 and beta at index m+1)
        L (float): Length of the pendulum
        T (float): Total simulation time
    """
    # Initialize Pygame
    pygame.init()
    
    # Screen setup
    WIDTH, HEIGHT = 800, 600
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Project 3 - Pendulum Animation")
    
    # Colors
    WHITE = (255, 255, 255)
    BLACK = (30, 30, 30)
    BLUE = (0, 120, 215)
    
    # Visual Scaling
    # L=1 meter translates to 350 pixels on the screen
    scale = 350 
    pivot = (WIDTH // 2, 100)
    
    # Timing calculation
    m_plus_1 = len(theta) - 1
    h = T / m_plus_1
    FPS = int(1 / h) if h > 0 else 60
    clock = pygame.time.Clock()
    
    running = True
    frame = 0
    
    while running:
        # Handle closing the window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
        screen.fill(WHITE)
        
        # Loop the animation if it reaches the end
        current_frame = frame % len(theta)
        current_theta = theta[current_frame]
        
        # 1. Physics to Pixel mapping
        # Pygame Y-axis increases downwards, so adding cosine puts it below the pivot
        x = pivot[0] + L * scale * math.sin(current_theta)
        y = pivot[1] + L * scale * math.cos(current_theta)
        
        # 2. Draw the Pendulum
        pygame.draw.line(screen, BLACK, pivot, (x, y), 3)             # Rod
        pygame.draw.circle(screen, BLACK, pivot, 6)                   # Pivot joint
        pygame.draw.circle(screen, BLUE, (int(x), int(y)), 20)        # Bob
        
        # 3. Draw HUD (Heads Up Display)
        current_time = current_frame * h
        font = pygame.font.SysFont("Arial", 24)
        time_text = font.render(f"Time: {current_time:.2f} s / {T:.2f} s", True, BLACK)
        angle_text = font.render(f"Theta: {current_theta:.2f} rad", True, BLACK)
        
        screen.blit(time_text, (20, 20))
        screen.blit(angle_text, (20, 50))
        
        pygame.display.flip()
        
        # Advance frame and lock framerate
        frame += 1
        clock.tick(FPS)

    pygame.quit()

# --- How to use it in your main() function ---
# After computing your final theta array using Newton-Raphson:
# full_theta = np.concatenate(([alpha], theta, [beta]))  <-- If your theta only has internal points
# animate_pendulum(full_theta, L, T)
