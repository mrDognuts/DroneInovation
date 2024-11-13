/* import Chart from 'chart.js/auto'; */


const chart = document.querySelector('#chart').getContext('2d');

new Chart(chart, {
  type: 'line',
  data: {
    labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
    datasets: [
      {
        label: 'Flights',
        data: [120, 135, 150, 140, 180, 175, 200, 195, 210, 185, 160, 155],
        borderColor: 'rgba(75, 192, 192, 1)',  // Line color
        backgroundColor: 'rgba(75, 192, 192, 0.2)', // Fill color
        borderWidth: 2,
        tension: 0.4  // Smooths out the line
      },
      {
        label: 'Claims',
        data: [10, 15, 12, 20, 18, 22, 17, 25, 30, 28, 24, 20],
        borderColor: 'rgba(255, 99, 132, 1)',
        backgroundColor: 'rgba(255, 99, 132, 0.2)',
        borderWidth: 2,
        tension: 0.4
      }
    ]
  },
  options: {
    responsive: true,
    plugins: {
      legend: {
        position: 'top',
      },
      labels:
      {
        color: '#fdf2e9'
      },
      tooltip: {
        enabled: true,
      },
      titleFont:
      {
        color: '#fdf2e9'
      },
    },
    scales: {
      y: {
        beginAtZero: true,
        title: {
          display: true,
          text: 'Number'
        }
      },
      x: {
        title: {
          display: true,
          text: 'Months'
        }
      }
    }
  }
});

// show or hide sidebar

const menuBtn = document.querySelector('#menu-btn');
const closeBtn = document.querySelector('#close');
const sidebar = document.querySelector('aside');

menuBtn.addEventListener('click',() =>
  {
     sidebar.style.display = 'block';
  })

  closeBtn.addEventListener('click',() =>
    { 
       sidebar.style.display = 'none';
    })

const themeBtn = document.querySelector('.themetoggle');

themeBtn.addEventListener('click',() =>
  {
    document.body.classList.toggle('dark-mode');

    themeBtn.querySelector('span:first-child').classList.toggle('active');
    themeBtn.querySelector('span:last-child').classList.toggle('active');
  })