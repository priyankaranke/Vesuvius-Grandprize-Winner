import os
import cv2


def run_sanity_checks():
    for fragment_id in ['20231210121321', '20231106155350', '20231005123336', '20230820203112', '20230620230619', '20230826170124', '20230702185753', '20230522215721', '20230531193658', '20230520175435', '20230903193206', '20230902141231', '20231007101615', '20230929220924', 'recto', 'verso', '20231016151000', '20231012184423', '20231031143850']:
        fragment_id_ = "_".join(fragment_id.split(
            "_")[:min(1, len(fragment_id)-1)])
        print(fragment_id)
        if not os.path.exists(f'train_scrolls/{fragment_id_}'):
            fragment_id_ += "_superseded"
        assert os.path.exists(f'train_scrolls/{fragment_id_}/layers/00.tif') or os.path.exists(
            f'train_scrolls/{fragment_id_}/layers/00.jpg'), f"Fragment id {fragment_id_} has no surface volume"
        assert os.path.exists(
            f'train_scrolls/{fragment_id_}/{fragment_id}_inklabels.png')
        assert os.path.exists(
            f'train_scrolls/{fragment_id_}/{fragment_id}_mask.png')
    assert os.path.exists(f'train_scrolls/20231022170901/layers/00.tif')
    assert os.path.exists(
        f'train_scrolls/20231022170901/20231022170901_inklabels.tiff')
    assert os.path.exists(
        f'train_scrolls/20231022170901/20231022170901_mask.png')


def prepare_data():
    for l in os.listdir('all_labels/'):
        if '.png' in l:
            f_id = l[:-14]
            f_id_ = f_id
            if not os.path.exists(f'train_scrolls/{f_id}'):
                f_id_ = f_id + "_superseded"
            if os.path.exists(f'train_scrolls/{f_id_}'):
                img = cv2.imread(f'all_labels/{f_id}_inklabels.png', 0)
                cv2.imwrite(f"train_scrolls/{f_id_}/{f_id}_inklabels.png", img)
            else:
                print(f"couldnt find {f_id_}")
        if '.tiff' in l:
            f_id = l[:-15]
            f_id_ = f_id
            if not os.path.exists(f'train_scrolls/{f_id}'):
                f_id_ = f_id + "_superseded"
            if os.path.exists(f'train_scrolls/{f_id_}'):
                img = cv2.imread(f'all_labels/{f_id}_inklabels.tiff', 0)
                cv2.imwrite(
                    f"train_scrolls/{f_id_}/{f_id}_inklabels.tiff", img)
            else:
                print(f"couldnt find {f_id_}")


if __name__ == "__main__":
    prepare_data()
    run_sanity_checks()
