from options import Options
from apps import App

if __name__ == '__main__':
    opt = Options().parse()

    app = App(opt)
    app.run()