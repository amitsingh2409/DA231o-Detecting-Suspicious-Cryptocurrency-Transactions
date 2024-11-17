import os
import shutil
import subprocess
import signal
import argparse
import configparser
import socket

ip_address = socket.gethostbyname(socket.gethostname())


def update_airflow_config():
    config = configparser.ConfigParser()
    config.read("airflow/airflow.cfg")
    config["core"]["load_examples"] = "False"
    with open("airflow/airflow.cfg", "w") as configfile:
        config.write(configfile)


def stop_service(pid_file):
    if os.path.exists(pid_file):
        with open(pid_file, "r") as f:
            pid = int(f.read().strip())
            try:
                os.kill(pid, signal.SIGTERM)
                os.remove(pid_file)
            except ProcessLookupError:
                print(f"Process with PID {pid} not found.")


def stop_service_on_port(port):
    result = subprocess.run(["lsof", "-i", f":{port}"], capture_output=True, text=True)
    for line in result.stdout.splitlines()[1:]:
        parts = line.split()
        pid = int(parts[1])
        try:
            os.kill(pid, signal.SIGTERM)
            print(f"Stopped process with PID {pid} on port {port}")
        except ProcessLookupError:
            print(f"Process with PID {pid} not found.")


def setup_mlflow(mlflow_port):
    stop_service("pids/mlflow.pid")
    stop_service_on_port(mlflow_port)
    if os.path.exists("db/mlflow.db"):
        os.remove("db/mlflow.db")
    mlflow_proc = subprocess.Popen(
        [
            "mlflow",
            "ui",
            "--backend-store-uri",
            "sqlite:///db/mlflow.db",
            "--host",
            "0.0.0.0",
            "--port",
            str(mlflow_port),
        ],
        stdout=open("logs/mlflow.log", "w"),
        stderr=subprocess.STDOUT,
    )
    with open("pids/mlflow.pid", "w") as f:
        f.write(str(mlflow_proc.pid))
    os.environ["MLFLOW_TRACKING_URI"] = f"http://{ip_address}:{mlflow_port}"


def setup_airflow(airflow_web_port):
    stop_service("pids/airflow_webserver.pid")
    stop_service("pids/airflow_scheduler.pid")
    stop_service_on_port(airflow_web_port)
    airflow_home = os.path.join(os.getcwd(), "airflow")
    os.environ["AIRFLOW_HOME"] = airflow_home
    os.environ["PYTHONWARNINGS"] = "ignore"
    if os.path.exists(airflow_home):
        shutil.rmtree(airflow_home, ignore_errors=True)
    subprocess.run(["airflow", "db", "init"], check=True)
    subprocess.run(
        [
            "airflow",
            "users",
            "create",
            "--username",
            "admin",
            "--firstname",
            "Amit",
            "--lastname",
            "Singh",
            "--role",
            "Admin",
            "--email",
            "amitsingh@iisc.ac.in",
            "--password",
            "iisc@123",
        ],
        check=True,
    )
    update_airflow_config()
    airflow_web_proc = subprocess.Popen(
        ["airflow", "webserver", "--port", str(airflow_web_port), "--host", "0.0.0.0"],
        env=os.environ,
        stdout=open("logs/airflow_webserver.log", "w"),
        stderr=subprocess.STDOUT,
    )
    with open("pids/airflow_webserver.pid", "w") as f:
        f.write(str(airflow_web_proc.pid))
    airflow_scheduler_proc = subprocess.Popen(
        ["airflow", "scheduler"],
        env=os.environ,
        stdout=open("logs/airflow_scheduler.log", "w"),
        stderr=subprocess.STDOUT,
    )
    with open("pids/airflow_scheduler.pid", "w") as f:
        f.write(str(airflow_scheduler_proc.pid))


def copy_dag_file():
    airflow_home = os.environ["AIRFLOW_HOME"]
    dags_folder = os.path.join(airflow_home, "dags")
    if not os.path.exists(dags_folder):
        os.makedirs(dags_folder)
    shutil.copy("ml_pipeline_dag.py", dags_folder)


def main(mlflow_port, airflow_web_port):
    for folder in ["db", "pids", "logs"]:
        if not os.path.exists(folder):
            os.makedirs(folder)
    setup_mlflow(mlflow_port)
    setup_airflow(airflow_web_port)
    copy_dag_file()
    print(
        "Setup complete. Access MLflow UI at http://0.0.0.0:{} and Airflow UI at http://0.0.0.0:{}".format(
            mlflow_port, airflow_web_port
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Setup MLflow and Airflow")
    parser.add_argument("--mlflow-port", type=int, default=5000, help="Port for MLflow UI")
    parser.add_argument(
        "--airflow-web-port", type=int, default=8080, help="Port for Airflow Web UI"
    )
    parser.add_argument("--stop", action="store_true", help="Stop MLflow and Airflow services")
    args = parser.parse_args()
    if args.stop:
        stop_service("pids/mlflow.pid")
        stop_service("pids/airflow_webserver.pid")
        stop_service("pids/airflow_scheduler.pid")
        stop_service_on_port(args.mlflow_port)
        stop_service_on_port(args.airflow_web_port)
        print("Stopped MLflow and Airflow services.")
    else:
        main(args.mlflow_port, args.airflow_web_port)
