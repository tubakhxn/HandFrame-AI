

STYLES = [
    {
        "name": "Thermal / Heatmap",
        "prompt": "thermal infrared heatmap portrait, glowing purple and blue "
                   "grid overlay, futuristic scan effect, high contrast",
    },
    {
        "name": "Anime",
        "prompt": "clean anime illustration portrait, soft cel shading, "
                   "studio ghibli inspired lighting, pastel sky background",
    },
    {
        "name": "Classic Oil Painting",
        "prompt": "renaissance oil painting portrait, rembrandt chiaroscuro "
                   "lighting, rich dark background, museum quality brushwork",
    },
    {
        "name": "Pop Art",
        "prompt": "vibrant pop art portrait, bold geometric shapes, "
                   "andy warhol inspired color blocking, halftone texture",
    },
    {
        "name": "Psychedelic Swirl",
        "prompt": "psychedelic rainbow swirl portrait, kaleidoscopic color "
                   "noise background, trippy poster art, high saturation",
    },
    {
        "name": "Watercolor Sketch",
        "prompt": "delicate watercolor portrait sketch, soft pastel washes, "
                   "loose ink linework, paper texture background",
    },
    {
        "name": "Cyberpunk Neon",
        "prompt": "cyberpunk neon city portrait, glowing magenta and cyan "
                   "signage reflections, rain-soaked street background, "
                   "blade runner aesthetic",
    },
    {
        "name": "Van Gogh",
        "prompt": "post-impressionist portrait in the style of van gogh, "
                   "swirling thick brushstrokes, vivid starry night palette",
    },
    {
        "name": "Graffiti",
        "prompt": "urban graffiti mural style portrait, spray paint texture, "
                   "bold outlines, street art color palette",
    },
    {
        "name": "Pencil Sketch",
        "prompt": "graphite pencil sketch portrait, fine crosshatching, "
                   "realistic shading, white paper background",
    },
    {
        "name": "Pixel Art",
        "prompt": "16-bit pixel art portrait, limited color palette, "
                   "retro video game character style",
    },
    {
        "name": "Comic Book",
        "prompt": "comic book ink portrait, bold black outlines, ben-day dot "
                   "shading, dramatic action-panel lighting",
    },
    {
        "name": "Low Poly",
        "prompt": "low poly 3D portrait, faceted geometric shading, "
                   "flat color gradients, modern minimalist art",
    },
    {
        "name": "Stained Glass",
        "prompt": "stained glass window portrait, leaded black outlines, "
                   "jewel-toned translucent panels, cathedral lighting",
    },
    {
        "name": "Charcoal",
        "prompt": "expressive charcoal drawing portrait, smudged dramatic "
                   "shadows, textured paper, high contrast monochrome",
    },
    {
        "name": "Vaporwave",
        "prompt": "vaporwave aesthetic portrait, pastel pink and teal "
                   "gradient, retro 80s grid background, glitch accents",
    },
    {
        "name": "Studio Ghibli Sky",
        "prompt": "soft hand-painted anime portrait, dreamy cloud sky "
                   "background, warm golden hour lighting, gentle color palette",
    },
]


def get_style(index):
    return STYLES[index % len(STYLES)]


def style_count():
    return len(STYLES)
