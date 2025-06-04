# News

We are happy to announce that **VPI** has been accepted to IROS2024! 😆🎉🎉

Initial code released.

# Visual Preference Inference: An Image Sequence-Based Preference Reasoning in Tabletop Object Manipulation

[ [Project Page](https://joonhyung-lee.github.io/vpi/) | [Paper](https://arxiv.org/abs/2403.11513) | [Video](https://youtu.be/wUerBZhHhuU) ]

Official Implementation of the paper ***Visual Preference Inference: An Image Sequence-Based Preference Reasoning in Tabletop Object Manipulation***

![fig_overview](https://github.com/joonhyung-lee/vpi/raw/github-page/assets/images/fig-overview.png)

## How to Start?

> [!Note]
> At the time of writing this paper, we used the `gpt-4-vision-preview` API for our implementation. However, this model is now deprecated and has been replaced with `gpt-4o` or `gpt-4o-mini`. 

* Install requirements via `requirements.txt`
  ```bash
    pip3 install -r requirements.txt
  ```
* Add your OpenAI API key to use GPT-4V: [Here](https://github.com/joonhyung-lee/vpi/blob/main/key/my_key.txt)
  ```bash
    export OPENAI_API_KEY="your-api-key"
  ```
* Run the example notebook [`scripts/household-chain-of-visual-residuals.ipynb`](https://github.com/joonhyung-lee/vpi/blob/main/scripts/household-chain-of-visual-residuals.ipynb) to see how `VPI` works with household objects