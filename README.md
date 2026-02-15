# BitBlind

## Summary
Audio/Visual tool using Adversarial Machine Learning evasion attacks to mask online media from AI-based data scraping

## Overview
Adversarial machine learning techniques can be used to perturb data in ways that are imperceptible to humans but evade machine learning inference. Using these algorithms to mask the audio and video channels of an online video makes the running of voice-to-text and optical character recognition (OCR) useless while preserving the original data. Online platforms (YouTube, Instagram, TikTok) use these algorithms to extract data from posted content, which forms advertising predictions and even censors content. The censorship policies of these companies are often quite naive and unfair. For example, using the words “gun”, “die”, or even “YouTube” in a TikTok video can get an account shadow-banned even if the context is completely informative or otherwise benign. BitBlind is a tool that takes an input video and applies invisible perturbations to avoid platform-based data scraping and censorship.