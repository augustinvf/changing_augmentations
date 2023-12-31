import torch

def test_fct(device, model, test_dataloader, matrix, nb_experiences_by_class):
    model.eval()

    total_tests = 0
    positive_tests = 0

    for mini_batch, labels in test_dataloader :

        image_without_augmentation = mini_batch.to(device)
        labels = labels.to(device)

        with torch.no_grad() :
            y_hat = model(image_without_augmentation, "supervised")

        maj_confusion_matrix(matrix, y_hat, labels)
        nb_experiences_by_class += torch.bincount(labels)

        total_tests += labels.shape[0]
        positive_tests += torch.sum(torch.eq(torch.argmax(y_hat, axis = 1), labels))

    final_accuracy = positive_tests / total_tests

    matrix = matrix / nb_experiences_by_class.reshape(-1, 1)
    torch.save(matrix, "./confusion_matrix_sim_clr_1_100_100.pth")

    return final_accuracy

def maj_confusion_matrix(matrix, y_hat, labels):
        for index, label in enumerate(labels) :
            prediction = torch.argmax(y_hat[index, :]).detach().item()
            matrix[label,prediction] += 1
