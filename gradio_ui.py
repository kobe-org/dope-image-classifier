# from flask import Flask, request, session, jsonify, abort, send_file, render_template, redirect
import gradio as gr

def greet(name):
  return "Hello " + name + "!!"


def main():
    iface = gr.Interface(fn=greet, inputs="text", outputs="text")
    iface.launch()


if __name__ == '__main__':
    main()