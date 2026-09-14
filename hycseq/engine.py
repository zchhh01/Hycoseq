import random
import statistics
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, matthews_corrcoef

from .data import build_loaders
from .runtime import build_model, build_optimizer, load_checkpoint, save_checkpoint


def metric_name(settings):
    return "Macro-F1" if settings.corpus == "GUE" and settings.task.lower() in {"virus_covid", "covid"} else "MCC"


def metric_value(settings, truth, predicted):
    if metric_name(settings) == "Macro-F1":
        return float(f1_score(truth, predicted, average="macro", zero_division=0))
    return float(matthews_corrcoef(truth, predicted))


def synchronize(device):
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.synchronize()


def pass_over(model, loader, loss_function, settings, device, optimizer=None, epoch=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    truth = []
    predicted = []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch_index, (values, labels, lengths) in enumerate(loader):
            values = values.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)
            if training:
                optimizer.zero_grad()
            scores = model(values, lengths=lengths)
            loss = loss_function(scores, labels)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * values.size(0)
            choices = scores.argmax(dim=1)
            correct += choices.eq(labels).sum().item()
            truth.append(labels.detach().cpu().numpy())
            predicted.append(choices.detach().cpu().numpy())
            if training and batch_index % 10 == 0:
                print("epoch={} batch={}/{} loss={:.6f}".format(epoch, batch_index, len(loader), loss.item()))
    truth_array = np.concatenate(truth)
    predicted_array = np.concatenate(predicted)
    return {
        "loss": total_loss / len(loader.dataset),
        "accuracy": correct / len(loader.dataset),
        "score": metric_value(settings, truth_array, predicted_array),
    }


def report(split, result, settings):
    print("{} loss={:.6f} accuracy={:.2%} {}={:.6f}".format(split, result["loss"], result["accuracy"], metric_name(settings), result["score"]))


def run_experiment(settings):
    random.seed(settings.seed)
    np.random.seed(settings.seed)
    torch.manual_seed(settings.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(settings.seed)
    device = torch.device(settings.accelerator if torch.cuda.is_available() or not settings.accelerator.startswith("cuda") else "cpu")
    train_loader, valid_loader, test_loader = build_loaders(settings)
    model = build_model(settings).to(device)
    optimizer, scheduler = build_optimizer(model, settings)
    loss_function = nn.CrossEntropyLoss()
    if settings.resume_from:
        load_checkpoint(settings.resume_from, model, optimizer, scheduler)
    parameter_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print("run={} device={} trainable_parameters={}".format(settings.run_id, device, parameter_count))
    best_score = -float("inf")
    best_epoch = -1
    durations = []
    save_root = Path(settings.save_root) if settings.save_root else None
    best_path = save_root / (settings.run_id + ".best.pt") if save_root else None
    for epoch in range(settings.epochs):
        synchronize(device)
        started = time.perf_counter()
        train_result = pass_over(model, train_loader, loss_function, settings, device, optimizer, epoch)
        synchronize(device)
        durations.append(time.perf_counter() - started)
        valid_result = pass_over(model, valid_loader, loss_function, settings, device)
        report("train", train_result, settings)
        report("validation", valid_result, settings)
        if valid_result["score"] > best_score:
            best_score = valid_result["score"]
            best_epoch = epoch
            if best_path:
                save_checkpoint(best_path, model, optimizer, scheduler, settings, epoch, best_score)
        if scheduler is not None:
            if epoch + 1 in settings.schedule_steps:
                for group in optimizer.param_groups:
                    if group.get("name") == "manifold":
                        group["lr"] /= settings.schedule_factor
            scheduler.step()
    final_path = save_root / (settings.run_id + ".final.pt") if save_root else None
    if final_path:
        save_checkpoint(final_path, model, optimizer, scheduler, settings, settings.epochs - 1, best_score)
    if best_path and best_path.is_file():
        load_checkpoint(best_path, model)
    test_result = pass_over(model, test_loader, loss_function, settings, device)
    report("test", test_result, settings)
    print("best_epoch={} best_validation_{}={:.6f}".format(best_epoch, metric_name(settings), best_score))
    print("training_seconds total={:.2f} mean={:.2f} median={:.2f}".format(sum(durations), statistics.mean(durations), statistics.median(durations)))
    return test_result
