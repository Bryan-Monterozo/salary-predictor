from salary_predictor.web import create_app


def main() -> None:
    app = create_app({"EXPERIMENTAL_MODE": True})
    app.run(host="127.0.0.1", port=5000, debug=False)


if __name__ == "__main__":
    main()
