import os
import pygame
import numpy as np

SCREEN_WIDTH, SCREEN_HEIGHT, FPS = 288, 512, 60
GROUND_Y = 400
GROUND_HEIGHT = SCREEN_HEIGHT - GROUND_Y

# Bird Physics & Size
BIRD_X, BIRD_WIDTH, BIRD_HEIGHT = 50, 30, 22
GRAVITY, FLAP_VELOCITY, MAX_VELOCITY = 0.8, -8.0, 10.0

# Pipe Configuration
PIPE_SPEED, PIPE_WIDTH, PIPE_GAP, PIPE_SPACING = 3.0, 52, 100, 160
MIN_GAP_TOP, MAX_GAP_TOP = 50, GROUND_Y - PIPE_GAP - 50

# Rewards
REWARD_SURVIVE, REWARD_PIPE, REWARD_DEATH = 0.1, 1.0, -1.0

# Colors
COLOR_SKY = (113, 197, 207)
COLOR_GROUND = (222, 216, 149)
COLOR_GRASS, COLOR_LINE = (115, 191, 46), (83, 56, 71)
COLOR_PIPE, COLOR_PIPE_DARK = (115, 191, 46), (90, 150, 35)
COLOR_BIRD, COLOR_BEAK = (250, 200, 30), (240, 90, 35)
COLOR_WHITE, COLOR_BLACK = (255, 255, 255), (0, 0, 0)


class FlappyEnv:

    def __init__(self, render_mode=None, seed=None):
        self.render_mode = render_mode
        self.rng = np.random.default_rng(seed)
        self.screen = self.font = self.big_font = None

        if self.render_mode == "human":
            if os.environ.get("SDL_VIDEODRIVER") == "dummy":
                del os.environ["SDL_VIDEODRIVER"]
            pygame.init()
            pygame.font.init()
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("Flappy Bird")
            self.font = pygame.font.SysFont("Arial", 18, bold=True)
            self.big_font = pygame.font.SysFont("Arial", 32, bold=True)
        else:
            os.environ["SDL_VIDEODRIVER"] = "dummy"

        self.reset()

    def _sample_gap(self):
        return int(self.rng.integers(MIN_GAP_TOP, MAX_GAP_TOP + 1))

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.bird_y, self.bird_vel = float(SCREEN_HEIGHT // 2), 0.0
        self.score, self.done, self.ground_x = 0, False, 0.0
        self.pipes = [
            {"x": float(SCREEN_WIDTH + 100 + i * PIPE_SPACING), "gap_top": float(self._sample_gap()), "scored": False}
            for i in range(3)
        ]
        return self._get_obs()

    def _get_next_pipe(self):
        candidates = [p for p in self.pipes if p["x"] + PIPE_WIDTH >= BIRD_X]
        return min(candidates, key=lambda p: p["x"]) if candidates else self.pipes[0]

    def _get_obs(self):
        p = self._get_next_pipe()
        return np.array([
            self.bird_y / SCREEN_HEIGHT,
            self.bird_vel / MAX_VELOCITY,
            (p["x"] - BIRD_X) / SCREEN_WIDTH,
            p["gap_top"] / SCREEN_HEIGHT,
            (p["gap_top"] + PIPE_GAP) / SCREEN_HEIGHT,
        ], dtype=np.float32)

    def step(self, action):
        if self.done:
            return self._get_obs(), 0.0, True, {"score": self.score}

        if action == 1:
            self.bird_vel = FLAP_VELOCITY

        # Physics & scrolling
        self.bird_vel = min(self.bird_vel + GRAVITY, MAX_VELOCITY)
        self.bird_y += self.bird_vel
        for p in self.pipes:
            p["x"] -= PIPE_SPEED
        self.ground_x = (self.ground_x - PIPE_SPEED) % 16

        # Score passing
        passed = False
        for p in self.pipes:
            if not p["scored"] and (p["x"] + PIPE_WIDTH < BIRD_X):
                p["scored"] = True
                self.score += 1
                passed = True

        # Collisions
        if self.bird_y + BIRD_HEIGHT >= GROUND_Y:
            self.bird_y, self.done = float(GROUND_Y - BIRD_HEIGHT), True
        elif self.bird_y <= 0:
            self.bird_y, self.done = 0.0, True
        else:
            bird_rect = pygame.Rect(BIRD_X, int(self.bird_y), BIRD_WIDTH, BIRD_HEIGHT)
            for p in self.pipes:
                px, gt = int(p["x"]), int(p["gap_top"])
                upper = pygame.Rect(px, 0, PIPE_WIDTH, gt)
                lower = pygame.Rect(px, gt + PIPE_GAP, PIPE_WIDTH, GROUND_Y - (gt + PIPE_GAP))
                if bird_rect.colliderect(upper) or bird_rect.colliderect(lower):
                    self.done = True
                    break

        reward = REWARD_DEATH if self.done else (REWARD_SURVIVE + (REWARD_PIPE if passed else 0.0))

        # Recycle pipes
        for p in self.pipes:
            if p["x"] < -PIPE_WIDTH:
                p["x"] = max(pipe["x"] for pipe in self.pipes) + PIPE_SPACING
                p["gap_top"] = self._sample_gap()
                p["scored"] = False

        return self._get_obs(), reward, self.done, {"score": self.score}

    def _draw(self):
        self.screen.fill(COLOR_SKY)
        cap_h, cap_ov = 24, 3

        # Pipes
        for p in self.pipes:
            px, gt = int(p["x"]), int(p["gap_top"])
            gb = gt + PIPE_GAP
            # Upper pipe
            pygame.draw.rect(self.screen, COLOR_PIPE, (px, 0, PIPE_WIDTH, gt - cap_h))
            pygame.draw.rect(self.screen, COLOR_PIPE, (px - cap_ov, gt - cap_h, PIPE_WIDTH + 2 * cap_ov, cap_h))
            pygame.draw.rect(self.screen, COLOR_LINE, (px, 0, PIPE_WIDTH, gt - cap_h), 2)
            pygame.draw.rect(self.screen, COLOR_LINE, (px - cap_ov, gt - cap_h, PIPE_WIDTH + 2 * cap_ov, cap_h), 2)
            pygame.draw.line(self.screen, COLOR_PIPE_DARK, (px + PIPE_WIDTH - 6, 0), (px + PIPE_WIDTH - 6, gt - cap_h), 4)
            # Lower pipe
            pygame.draw.rect(self.screen, COLOR_PIPE, (px - cap_ov, gb, PIPE_WIDTH + 2 * cap_ov, cap_h))
            pygame.draw.rect(self.screen, COLOR_PIPE, (px, gb + cap_h, PIPE_WIDTH, GROUND_Y - (gb + cap_h)))
            pygame.draw.rect(self.screen, COLOR_LINE, (px - cap_ov, gb, PIPE_WIDTH + 2 * cap_ov, cap_h), 2)
            pygame.draw.rect(self.screen, COLOR_LINE, (px, gb + cap_h, PIPE_WIDTH, GROUND_Y - (gb + cap_h)), 2)
            pygame.draw.line(self.screen, COLOR_PIPE_DARK, (px + PIPE_WIDTH - 6, gb + cap_h), (px + PIPE_WIDTH - 6, GROUND_Y), 4)

        # Ground
        pygame.draw.rect(self.screen, COLOR_GROUND, (0, GROUND_Y, SCREEN_WIDTH, GROUND_HEIGHT))
        pygame.draw.rect(self.screen, COLOR_GRASS, (0, GROUND_Y, SCREEN_WIDTH, 14))
        pygame.draw.line(self.screen, COLOR_LINE, (0, GROUND_Y), (SCREEN_WIDTH, GROUND_Y), 2)
        pygame.draw.line(self.screen, COLOR_LINE, (0, GROUND_Y + 14), (SCREEN_WIDTH, GROUND_Y + 14), 2)
        for x in range(int(self.ground_x) - 16, SCREEN_WIDTH + 16, 16):
            pygame.draw.line(self.screen, COLOR_LINE, (x, GROUND_Y + 14), (x + 8, GROUND_Y), 2)

        # Bird
        bx, by = BIRD_X, int(self.bird_y)
        pygame.draw.ellipse(self.screen, COLOR_BIRD, (bx, by, BIRD_WIDTH, BIRD_HEIGHT))
        pygame.draw.ellipse(self.screen, COLOR_LINE, (bx, by, BIRD_WIDTH, BIRD_HEIGHT), 2)
        pygame.draw.ellipse(self.screen, (255, 235, 120), (bx + 3, by + 6, 14, 10))
        pygame.draw.circle(self.screen, COLOR_WHITE, (bx + 20, by + 6), 5)
        pygame.draw.circle(self.screen, COLOR_LINE, (bx + 20, by + 6), 5, 1)
        pygame.draw.circle(self.screen, COLOR_BLACK, (bx + 22, by + 6), 2)
        pygame.draw.polygon(self.screen, COLOR_BEAK, [(bx + 26, by + 8), (bx + 35, by + 12), (bx + 26, by + 16)])
        pygame.draw.polygon(self.screen, COLOR_LINE, [(bx + 26, by + 8), (bx + 35, by + 12), (bx + 26, by + 16)], 1)

        # Score
        score_text = str(self.score)
        sh = self.big_font.render(score_text, True, COLOR_BLACK)
        txt = self.big_font.render(score_text, True, COLOR_WHITE)
        sx = SCREEN_WIDTH // 2 - txt.get_width() // 2
        self.screen.blit(sh, (sx + 2, 32))
        self.screen.blit(txt, (sx, 30))

        # Game Over overlay
        if self.done:
            gosh = self.big_font.render("GAME OVER", True, COLOR_BLACK)
            gotxt = self.big_font.render("GAME OVER", True, (217, 83, 79))
            gx = SCREEN_WIDTH // 2 - gotxt.get_width() // 2
            self.screen.blit(gosh, (gx + 2, SCREEN_HEIGHT // 2 - 48))
            self.screen.blit(gotxt, (gx, SCREEN_HEIGHT // 2 - 50))

            subsh = self.font.render("Press SPACE to restart", True, COLOR_BLACK)
            subtxt = self.font.render("Press SPACE to restart", True, COLOR_WHITE)
            rx = SCREEN_WIDTH // 2 - subtxt.get_width() // 2
            self.screen.blit(subsh, (rx + 1, SCREEN_HEIGHT // 2 + 1))
            self.screen.blit(subtxt, (rx, SCREEN_HEIGHT // 2))

    def render(self):
        if self.render_mode != "human" or self.screen is None:
            return
        self._draw()
        pygame.display.flip()

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None


if __name__ == "__main__":
    env = FlappyEnv()
    for episode in range(1, 6):
        obs = env.reset()
        done = False
        while not done:
            action = int(env.rng.integers(0, 2))
            obs, reward, done, info = env.step(action)
        print(f"Episode {episode}: Score = {info['score']}")
    env.close()
