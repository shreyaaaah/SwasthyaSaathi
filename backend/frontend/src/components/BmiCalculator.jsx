import React, { useId, useState } from 'react';
import { Calculator, RotateCcw } from 'lucide-react';

const inputClassName = 'w-full rounded-lg border border-slate-600 bg-slate-950/80 px-3 py-2 text-sm text-slate-100 placeholder:text-slate-500 focus:border-teal-400 focus:outline-none focus:ring-2 focus:ring-teal-500/25';

function BmiCalculator({ showTitle = true }) {
  const inputId = useId();
  const [heightCm, setHeightCm] = useState('');
  const [weightKg, setWeightKg] = useState('');
  const [ageYears, setAgeYears] = useState('');

  const hasAnyInput = Boolean(heightCm || weightKg || ageYears);
  let message = 'Enter height, weight, and age to calculate BMI.';
  let bmi = null;
  let category = null;
  let categoryColor = 'text-slate-300';

  if (hasAnyInput && (!heightCm || !weightKg || !ageYears)) {
    message = 'Fill in all three measurements to see your result.';
  } else if (hasAnyInput) {
    const height = Number(heightCm);
    const weight = Number(weightKg);
    const age = Number(ageYears);

    if (!Number.isFinite(height) || !Number.isFinite(weight) || !Number.isInteger(age)
      || height < 80 || height > 250 || weight < 2 || weight > 350 || age < 2 || age > 120) {
      message = 'Use height 80–250 cm, weight 2–350 kg, and age 2–120 in whole years.';
    } else {
      bmi = weight / ((height / 100) ** 2);
      if (age < 20) {
        message = 'For ages 2–19, BMI category uses age- and sex-specific percentiles. This calculator does not assign an adult category.';
      } else if (bmi < 18.5) {
        category = 'Underweight';
        categoryColor = 'text-amber-300';
        message = 'Adult BMI range: below 18.5.';
      } else if (bmi < 25) {
        category = 'Normal / healthy weight';
        categoryColor = 'text-emerald-300';
        message = 'Adult BMI range: 18.5 to under 25.';
      } else if (bmi < 30) {
        category = 'Overweight';
        categoryColor = 'text-amber-300';
        message = 'Adult BMI range: 25 to under 30.';
      } else {
        category = 'Obesity range';
        categoryColor = 'text-rose-300';
        message = 'Adult BMI range: 30 or higher.';
      }
    }
  }

  const clearValues = () => {
    setHeightCm('');
    setWeightKg('');
    setAgeYears('');
  };

  return (
    <section aria-label={showTitle ? undefined : 'BMI calculator'} aria-labelledby={showTitle ? `${inputId}-title` : undefined} className="rounded-2xl border border-slate-700/80 bg-slate-900/80 p-4 shadow-xl">
      {(showTitle || hasAnyInput) && <div className="mb-3 flex items-center justify-between gap-2">
        {showTitle ? <h2 id={`${inputId}-title`} className="flex items-center gap-2 text-sm font-semibold text-slate-100">
          <Calculator aria-hidden="true" className="h-4 w-4 text-teal-300" />
          BMI calculator
        </h2> : <span />}
        {hasAnyInput && (
          <button type="button" onClick={clearValues} aria-label="Clear BMI values" className="rounded-md p-1.5 text-slate-400 hover:bg-slate-800 hover:text-teal-200">
            <RotateCcw aria-hidden="true" className="h-3.5 w-3.5" />
          </button>
        )}
      </div>}

      <div className="space-y-3">
        <label htmlFor={`${inputId}-height`} className="block text-xs text-slate-300">
          Height <span className="text-slate-500">(cm)</span>
          <input id={`${inputId}-height`} className={`${inputClassName} mt-1`} type="number" inputMode="decimal" min="80" max="250" step="0.1" value={heightCm} onChange={(event) => setHeightCm(event.target.value)} placeholder="e.g. 165" />
        </label>
        <label htmlFor={`${inputId}-weight`} className="block text-xs text-slate-300">
          Weight <span className="text-slate-500">(kg)</span>
          <input id={`${inputId}-weight`} className={`${inputClassName} mt-1`} type="number" inputMode="decimal" min="2" max="350" step="0.1" value={weightKg} onChange={(event) => setWeightKg(event.target.value)} placeholder="e.g. 60" />
        </label>
        <label htmlFor={`${inputId}-age`} className="block text-xs text-slate-300">
          Age <span className="text-slate-500">(years)</span>
          <input id={`${inputId}-age`} className={`${inputClassName} mt-1`} type="number" inputMode="numeric" min="2" max="120" step="1" value={ageYears} onChange={(event) => setAgeYears(event.target.value)} placeholder="e.g. 25" />
        </label>
      </div>

      <div aria-live="polite" className="mt-4 rounded-xl border border-slate-700 bg-slate-950/70 p-3">
        {bmi === null ? (
          <p className="text-xs leading-relaxed text-slate-400">{message}</p>
        ) : (
          <>
            <p className="text-xs text-slate-400">Your BMI</p>
            <p className="my-1 text-2xl font-semibold tracking-tight text-slate-50">{bmi.toFixed(1)}</p>
            {category ? <p className={`text-sm font-semibold ${categoryColor}`}>{category}</p> : <p className="text-xs leading-relaxed text-slate-300">{message}</p>}
            {category && <p className="mt-1 text-[11px] text-slate-500">{message}</p>}
          </>
        )}
      </div>

      <p className="mt-3 text-[11px] leading-relaxed text-slate-500">BMI is a screening measure, not a diagnosis. Use only as general information.</p>
    </section>
  );
}

export default BmiCalculator;
