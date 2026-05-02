# ➤ AI Image Generator
elif menu == "AI Image Generator":
    st.subheader("🎨 AI Image Generator")

    prompt = st.text_input("Enter image prompt")

    if st.button("Generate Image"):
        if prompt:
            with st.spinner("Generating image... (first time may take time)"):
                from diffusers import StableDiffusionPipeline
                import torch

                # Load model (cache to avoid reloading every time)
                @st.cache_resource
                def load_model():
                    pipe = StableDiffusionPipeline.from_pretrained(
                        "runwayml/stable-diffusion-v1-5"
                    )
                    return pipe.to("cpu")

                pipe = load_model()

                image = pipe(prompt).images[0]

                st.image(image, caption="Generated Image", use_container_width=True)

                # Save option
                image.save("generated.png")
                st.success("Image generated and saved as generated.png")
        else:
            st.warning("Please enter a prompt")
