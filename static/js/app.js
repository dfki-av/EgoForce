// JavaScript to handle mouseover and mouseout events
var activeMethodPill = null;
var activeScenePill = null;
var activeModePill = null;
var activeVidID = 0;
var select = false;


$(document).ready(function () {
    var editor = CodeMirror.fromTextArea(document.getElementById("bibtex"), {
        lineNumbers: false,
        lineWrapping: true,
        readOnly: true
    });
    // 設定固定高度，例如 150px（可根據需要調整）
    editor.setSize(null, "180px");
    
    $(function () {
        $('[data-toggle="tooltip"]').tooltip()
    });

    activeMethodPill = $('.method-pill').filter('.active')[0];
    activeModePill = $('.mode-pill').filter('.active')[0];
    activeScenePill = $('.scene-pill').filter('.active')[0];

    console.log("console activae method pill: ", activeMethodPill)
    console.log("console activae mode pill: ", activeModePill)
    console.log("console activae scene pill: ", activeScenePill)

    // resizeAndPlay($('#sparsity')[0]);
});

function selectCompVideo(methodPill, scenePill, n_views, modePill) {
    // Your existing logic for video selection
    // var video = document.getElementById("compVideo");
    select = true;
    var videoSwitch = document.getElementById("compVideoSwitch");
    var viewNum = document.getElementById("compVideoValue");

    console.log("activate scene pill: ", activeScenePill)
    console.log("activate method pill: ", activeMethodPill)
    console.log("activate mode pill: ", activeModePill)

    // console.log("scene pill: ", scenePill)
    // console.log("method pill: ", methodPill)

    // console.log("method: ", methodPill.getAttribute("data-value"))
    // console.log("pill: ", scenePill.getAttribute("data-value"))
    // console.log("mode: ", modePill.getAttribute("data-value"))

    // console.log("active method pill 2: ", activeMethodPill.classList)

    if (activeMethodPill) {
        activeMethodPill.classList.remove("active");
    }
    if (activeScenePill) {
        activeScenePill.classList.remove("active");
    }
    if (modePill) {
        activeModePill.classList.remove("active");
        modePill.classList.add("active");
        activeModePill = modePill;
    }
    activeMethodPill = methodPill;
    activeScenePill = scenePill;
    methodPill.classList.add("active");
    scenePill.classList.add("active");
    method = methodPill.getAttribute("data-value");
    pill = scenePill.getAttribute("data-value");
    mode = activeModePill.getAttribute("data-value");


    

    // if (videoSwitch.checked) {
    //     mode = 'depth'
    // } else {
    //     mode = 'rgb'
    // }

    // swap video to avoid flickering
    activeVidID = 1 - activeVidID;
    var video_active = document.getElementById("compVideo" + activeVidID);
    var video_hidden = document.getElementById("compVideo" + (1 - activeVidID));
    // video_active.src = "./videos/comparison/" + pill + "_" + method + "_vs_ours_" + mode + ".mp4";
    video_active.src = "./videos/comparison/" + pill + "_" + method + "_vs_ours_" + mode + ".mp4";
    video_active.load();

    if (n_views) {
        viewNum.innerHTML = n_views;
    }
}


function updateMethodVisibility(mode) {
    // 定義每個mode可以顯示的方法列表
    const methodsByMode = {
        'rgb': ['gscream', 'gaussiangrouping', 'infusion', '2dgsleftrefill', 'spin'],
        'depth': ['infusion', 'poisson', 'gdd', 'scaleshift'],
        'mask': ['gaussiangrouping']
    };

    // 定義每個mode可以顯示的場景列表
    const scenesByMode = {
        'rgb': ['360USID_carton', '360USID_cone', '360USID_skateboard', '360USID_newcone', '360USID_sunflower', '360USID_plant', '360USID_cookie', 'Other360_kitchen', 'Other360_bear', 'Other360_bonsai', 'Other360_room', 'Other360_vasedeck', 'Other360_pinecone'], // 所有場景
        'depth': ['360USID_skateboard', '360USID_sunflower', 'Other360_bear', 'Other360_bonsai', 'Other360_vasedeck', 'Other360_pinecone'],
        'mask': ['360USID_carton', '360USID_cone', '360USID_skateboard', '360USID_newcone', '360USID_sunflower', '360USID_plant', '360USID_cookie', 'Other360_kitchen', 'Other360_bear', 'Other360_bonsai', 'Other360_room'],
    };

    const titlesByMode = {
        'rgb': 'AuraFusion360 outperforms other methods in unbounded 360° scene inpainting.',
        'depth': 'Our Adaptive Guided Depth Diffusion effectively produces well-aligned depth, enabling the accurate unprojection of the reference view into high-quality initial Gaussians. This ensures a reliable foundation for our subsequent SDEdit processes.',
        'mask': 'Our Depth-Aware Unseen Mask Generation scheme can accurately identify unseen regions.'
    };

    const descriptByMode = {
        'rgb': "Baseline method (left) vs AuraFusion360 (right).",
        'depth': "Baseline method (left) vs Our Adaptive Guided Depth Diffusion (right).",
        'mask': "Gaussian Grouping Video Tracker (left) vs Our Depth-Aware Unseen Mask Generation (right)."
    }

    const descriptionElement = document.getElementById('description-text');
    if (descriptionElement && descriptByMode[mode]) {
        descriptionElement.innerHTML = descriptByMode[mode];
    }

    // 更新標題
    const titleElement = document.getElementById('title-text');
    if (titleElement && titlesByMode[mode]) {
        titleElement.textContent = titlesByMode[mode];
    }

    // 更新方法按鈕的顯示
    const methodPills = document.querySelectorAll('.method-pill');
    methodPills.forEach(pill => {
        const methodValue = pill.getAttribute('data-value');
        if (methodsByMode[mode].includes(methodValue)) {
            pill.style.display = '';
        } else {
            pill.style.display = 'none';
            if (pill.classList.contains('active')) {
                const firstVisibleMethod = document.querySelector(`.method-pill[data-value="${methodsByMode[mode][0]}"]`);
                if (firstVisibleMethod) {
                    firstVisibleMethod.click();
                }
            }
        }
    });

    // 更新場景的顯示
    const scenePills = document.querySelectorAll('.scene-pill');
    scenePills.forEach(pill => {
        const sceneValue = pill.getAttribute('data-value');
        if (scenesByMode[mode].includes(sceneValue)) {
            pill.style.display = '';
        } else {
            pill.style.display = 'none';
            if (pill.classList.contains('active')) {
                const firstVisibleScene = document.querySelector(`.scene-pill[data-value="${scenesByMode[mode][0]}"]`);
                if (firstVisibleScene) {
                    firstVisibleScene.click();
                }
            }
        }
    });
}